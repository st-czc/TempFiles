# HIP/AMDGPU offload 流程分析 + 自定义 opt 插入

> 本文档梳理 clang 驱动下 HIP→AMDGPU 的编译+链接流程，并记录"在 device link 前插入自定义 `opt`"的方案。适用于较新的上游 llvm-project（clang 24 main 时代的新 offload 架构），无特定机器依赖。

## 0. 背景与范围

- 对象：上游 llvm-project 里，clang 驱动把 HIP 源码编译+链接成 AMDGPU 可执行文件的完整流程。
- 覆盖：默认（非 RDC）与 `-fgpu-rdc` 两条路径。
- 范围：编译+链接（不含运行时执行）。
- 结论性目标：在 `AMDGCN::Linker::constructLldCommand` 里插入 `llvm-link → opt → lld`，让自定义 pass 跑在**所有 device bitcode 合并之后、`ld.lld` 之前**。

## 1. 前置条件

- 一份较新的上游 llvm-project 构建，产物含 `clang`、`clang-offload-bundler`、`llvm-offload-binary`、`clang-linker-wrapper`、`ld.lld`、`opt`、`llvm-link`。
- 一份 ROCm 安装（`--rocm-path` 指向），提供：HIP 头（`include/hip`）、device bitcode（`amdgcn/bitcode`）、`libamdhip64.so`。
- 示例 arch：`gfx90a`（换其它 arch 同理）。

验证命令（占位符）：
```bash
CLANG=<llvm-build>/bin/clang++        # 或 PATH 里的 clang++
ROC=<rocm-path>
$CLANG -x hip --offload-arch=gfx90a --rocm-path=$ROC sample.cpp -o sample -v           # 默认
$CLANG -x hip --offload-arch=gfx90a --rocm-path=$ROC -fgpu-rdc sample.cpp -o sample_rdc -v  # RDC
```
`sample.cpp` 用最小 HIP 程序即可（`__global__ void k(int*)` + `hipMalloc` + `k<<<1,1>>>`）。

## 2. 关键结论：offload 流程（clang 24 main 新架构）

**重要**：clang 24（main）走的是新的 offload 架构（`llvm-offload-binary` + `clang-linker-wrapper` + 设备 LTO），不是老资料里"直接 `clang-offload-bundler`"的图。设备 triple 已是 `amdgpu9.0a-amd-amdhsa`（新 canonical），但 fatbin 的 bundle id 仍是 legacy `hip-amdgcn-amd-amdhsa--gfx90a`（为兼容 HIP runtime / CLR 的硬编码解析）。

### 默认（非 RDC）路径
1. device cc1：`-triple amdgpu9.0a-amd-amdhsa -emit-llvm-bc -flto=full -fcuda-is-device -mlink-builtin-bitcode ocml/ockl/oclc_* -target-cpu gfx90a -mllvm -amdgpu-internalize-symbols -include __clang_hip_runtime_wrapper.h` → device bitcode `.o`。
2. `llvm-offload-binary --image=...,triple=...,arch=gfx90a,kind=hip` 打包成 image。
3. `clang-linker-wrapper --emit-fatbin-only`：内部 device LTO（`ld.lld`）生成 amdgcn 代码对象 + `clang-offload-bundler` 打成 fatbin（`.hipfb`）。
4. host cc1：`-foffload-include-binary <fatbin>` 把 fatbin 嵌进 host 目标文件。
5. `clang-linker-wrapper` → host `ld` + `libamdhip64.so`。

### `-fgpu-rdc` 路径
1. device cc1：同上但**无** `-amdgpu-internalize-symbols`，带 `-fgpu-rdc`。
2. `llvm-offload-binary` 打包。
3. host cc1：`-fembed-offload-object=<image>`（只嵌原始 image，不生成 fatbin）。
4. `clang-linker-wrapper --wrapper-verbose` 展开：
   - 抽出 device 对象 → `clang --target=amdgpu9.0a-amd-amdhsa -flto=full` → `ld.lld --no-undefined -shared -plugin-opt=mcpu=gfx90a` → `.img` 代码对象。
   - `clang-offload-bundler -type=o -bundle-align=4096 -targets=host-...,hip-amdgcn-amd-amdhsa--gfx90a -input=/dev/null -input=...img -output=...hipfb`。
   - wrapper 对象嵌入 fatbin → host `ld` + `libamdhip64.so`。

### fatbin / offload bundler 内部
- bundle 头：magic `__CLANG_OFFLOAD_BUNDLE__` + `numCodeObjects` + 每条 `{offset, size, id[64]}` + 原始载荷。
- id 语法 `<kind>-<triple>[-<target-id>]`。实测 `clang-offload-bundler -list` 输出 `hip-amdgcn-amd-amdhsa--gfx90a` + `host-x86_64-unknown-linux-gnu-`。
- 嵌入段：`.hip_fatbin`（+ `.hipFatBinSegment`；RDC 另有 `llvm_offload_entries`、`.llvm.rodata.offloading`）。
- 运行时侧：`CGCUDANV.cpp` 生成 `.hip_fatbin` 段 + `HIPFatMagic=0x48495046` + 构造器调 `__hipRegisterFatBinary(&wrapper)`。

## 3. 自定义 opt 插入（已敲定方案）

**决定**：在 `clang/lib/Driver/ToolChains/HIPAMD.cpp` 的 `AMDGCN::Linker::constructLldCommand`（`HIPAMD.cpp:50-138`）里，恢复 `llvm-link → opt → lld` 三步（对应 `:191-193` 的旧注释 "llvm-link, opt, llc, then lld"）。RDC 时 `Inputs` 是**多个** device bitcode，先 `llvm-link` 合并再 `opt` 再喂 `lld`。

修改草图（插入在 `LldArgs` 拼好之后、`C.addCommand(lld)` 之前）：
```cpp
// 1) 合并所有输入 bitcode（同文件已有现成函数）
const char *MergedBC = HIP::getTempFile(C, "amdgpu-linked", "bc");
InputInfo MergedBCInfo(&JA, MergedBC, Output.getBaseInput());
constructLLVMLinkCommand(C, JA, Inputs, MergedBCInfo, Args);

// 2) 跑 custom opt
const char *OptBC = HIP::getTempFile(C, "amdgpu-opt", "bc");
InputInfo OptBCInfo(&JA, OptBC, Output.getBaseInput());
ArgStringList OptArgs{"-load-pass-plugin=/abs/path/libcustompasses.so",
                      "-passes=my-pass", MergedBC, "-o", OptBC};
C.addCommand(std::make_unique<Command>(JA, *this, ResponseFileSupport::None(),
                                       getToolChain().GetProgramPath("opt"),
                                       OptArgs, MergedBCInfo, OptBCInfo));

// 3) lld 只吃 OptBC（替换原来的 for-loop push Input.getFilename()）
LldArgs.append({"-o", Output.getFilename()});
LldArgs.push_back(OptBC);
```

**in-file 模板**：`constructLLVMLinkCommand`（`:36-48`）、`constructLinkAndEmitSpirvCommand`（`:145-189`，已是"先 link 出临时 `.bc` 再 addCommand 第二步"）。

### 外部 `.so`（`-load-pass-plugin`）结论
- **可以**。`-load-pass-plugin` 就是给 out-of-tree `.so` 用的（`dlopen` + `llvmGetPassPluginInfo()` 注册进 `PassBuilder`，`-passes=` 按名实例化）。
- 要求：`.so` 必须用**同一套 LLVM build** 编译（同 commit、同 `LLVM_ABI_BREAKING_CHECKS`/RTTI/异常配置）。用 CMake `add_llvm_pass_plugin(MyPass MyPass.cpp)`，`-DLLVM_DIR=<llvm-build>/lib/cmake/llvm`。
- 必须配 `-passes=my-pass`（或 `-passes=default<O3>,my-pass`）才会跑。
- 顶层 `-mllvm` 会**同时进 host 编译**（host cc1 也会 `dlopen` 并跑 pass），所以 device-only 场景用 `constructLldCommand` 硬编码更干净。
- 已实测 `-mllvm -time-passes` 确实透传到 `ld.lld` 命令行。

## 4. 关键 file:line 索引

| 位置 | 作用 |
|---|---|
| `clang/lib/Driver/Driver.cpp:994-1094` | `--offload-arch` 解析、HIP 默认 triple |
| `clang/lib/Driver/Driver.cpp:4118-4376` | `BuildOffloadingActions`（host+N device 拆分、RDC/非 RDC 分支） |
| `clang/lib/Driver/ToolChains/AMDGPU.cpp:921-999` | device cc1 参数（`-fcuda-is-device`、internalize、设备库） |
| `clang/lib/Driver/ToolChains/AMDGPU.cpp:1130-1164` | 设备库顺序 ocml→ockl→oclc |
| `clang/lib/Driver/ToolChains/HIPAMD.cpp:50-138` | **`constructLldCommand`（本次修改点）** |
| `clang/lib/Driver/ToolChains/HIPAMD.cpp:94-97` | device link 的 `-mllvm` → `ld.lld -plugin-opt=` |
| `clang/lib/Driver/ToolChains/HIPUtility.cpp:53-111` | fatbin bundle 构造（`-type=o -bundle-align=4096`） |
| `clang/tools/clang-linker-wrapper/ClangLinkerWrapper.cpp:424-503` | `amdgcn::fatbinary`（重调 bundler） |
| `clang/tools/clang-linker-wrapper/ClangLinkerWrapper.cpp:507-619` | `generic::clang`（wrapper 重调 clang→lld；`-mllvm` 转发在 `:544-548`/`:588-591`） |
| `clang/lib/Driver/OffloadBundler.cpp:69-123` | bundle magic / code-object id 解析 |
| `clang/lib/CodeGen/CGCUDANV.cpp:897-1029` | `.hip_fatbin` 段、`HIPFatMagic`、`__hipRegisterFatBinary` |

## 5. 下一步 / 待办

1. 实现 `constructLldCommand` 的 `llvm-link → opt → lld` 修改，用 `-v` 验证应看到 `llvm-link`/`opt`/`ld.lld` 三段。
2. 用 `add_llvm_pass_plugin` 编一个最小 `libcustompasses.so`，先走 `-mllvm -load-pass-plugin=` 快速验证 pass 逻辑。
3. 决定 opt 的管线：`-passes=my-pass`（只跑你的）vs `-passes=default<O3>,my-pass`；以及 lld 是否重跑默认 LTO（可加 `-plugin-opt` 微调）。
4. 语义提醒：非 RDC 时 `Inputs` 通常单 TU；RDC 时多 TU（本次目标场景）。
