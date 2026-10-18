#!/usr/bin/env python3
"""Disable samples that require selected unavailable CUDA libraries."""

from pathlib import Path


# Add other unavailable CUDA imported targets here as needed.
CUDA_LIBRARIES_TO_DISABLE = [
    "CUDA::nvrtc",
    "CUDA::curand",
    "CUDA::cufft",
    "CUDA::cublas",
    "CUDA::nvjpeg",
    "CUDA::nppc",
]


CUDA_OPTIONS_TO_DISABLE = [
    "CUDA_FATBIN_FILE",
]


CUDA_HEADERS_TO_DISABLE = [
    "<omp.h>",
    "<cuda/barrier>",
    '"cudaEGL.h"',
    "<thrust/random.h>",
    "<thrust/host_vector.h>",
    "<cooperative_groups/reduce.h>",
]
CUDA_HEADERS_TO_DISABLE_PREFIX = [
    f"#include {header}" for header in CUDA_HEADERS_TO_DISABLE
]


CUDA_FUNCTIONS_TO_DISABLE = [
    "cudaLimitPersistingL2CacheSize",
    "texCubemap",
    "tex2DLayered",
]


SOURCE_FILE_SUFFIXES = [".cu", ".cpp", ".h"]


def main() -> None:
    samples_dir = Path(__file__).resolve().parent / "Samples"
    if not samples_dir.is_dir():
        raise SystemExit(f"Samples directory not found: {samples_dir}")

    cmake_files = sorted(samples_dir.rglob("CMakeLists.txt"))
    sample_dirs = {cmake_file.parent for cmake_file in cmake_files}

    disabled_samples: set[Path] = set()
    for cmake_file in cmake_files:
        cmake_text = cmake_file.read_text(encoding="utf-8")
        if any(library in cmake_text for library in CUDA_LIBRARIES_TO_DISABLE):
            disabled_samples.add(cmake_file.parent)
        elif any(library in cmake_text for library in CUDA_OPTIONS_TO_DISABLE):
            disabled_samples.add(cmake_file.parent)

    header_prefixes = tuple(CUDA_HEADERS_TO_DISABLE_PREFIX)
    for source_file in sorted(samples_dir.rglob("*")):
        if source_file.suffix not in SOURCE_FILE_SUFFIXES:
            continue
        source_text = source_file.read_text(encoding="utf-8")
        if not any(
            line.startswith(header_prefixes) for line in source_text.splitlines()
        ):
            continue
        for parent in source_file.parents:
            if parent in sample_dirs:
                disabled_samples.add(parent)
                break

    parent_to_samples: dict[Path, set[str]] = {}
    for sample_dir in disabled_samples:
        parent_cmake = sample_dir.parent / "CMakeLists.txt"
        parent_to_samples.setdefault(parent_cmake, set()).add(sample_dir.name)

    for parent_cmake, sample_names in sorted(parent_to_samples.items()):
        if not parent_cmake.is_file():
            print(f"Parent CMakeLists.txt not found: {parent_cmake}")
            continue

        original = parent_cmake.read_text(encoding="utf-8")
        updated_lines = []
        changed = False
        for line in original.splitlines(keepends=True):
            for sample_name in sample_names:
                if line.startswith(f"add_subdirectory({sample_name})"):
                    line = "# " + line
                    changed = True
                    break
            updated_lines.append(line)
        updated = "".join(updated_lines)

        if changed:
            parent_cmake.write_text(updated, encoding="utf-8")
            print(f"Updated {parent_cmake}")


if __name__ == "__main__":
    main()
