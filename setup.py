#!/usr/bin/env python3
"""Compile the C++ and CUDA extensions for torch_fast_box_ops."""
from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

if __name__ == "__main__":
    ignore_files = {"profile.cpp"}
    targets = list(
        filter(
            lambda f: f.is_file()
            and f.suffix in {".cpp", ".cu"}
            and f.name not in ignore_files,
            Path("torch_fast_box_ops").iterdir(),
        )
    )
    setup(
        ext_modules=[
            CUDAExtension(
                "torch_fast_box_ops",
                targets,
                extra_compile_args={
                    "cxx": ["-O3", "-std=c++17"],
                    "nvcc": ["-O3", "-std=c++17"],
                },
            )
        ],
        cmdclass={"build_ext": BuildExtension},
    )
