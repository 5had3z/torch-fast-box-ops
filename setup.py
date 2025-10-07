#!/usr/bin/env python3
"""Compile the C++ and CUDA extensions for torch_fast_box_ops."""
from pathlib import Path

from setuptools import find_packages, setup
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
        packages=find_packages(),
        ext_modules=[
            CUDAExtension(
                "torch_fast_box_ops._tfbo",
                targets,
                extra_compile_args={
                    "cxx": ["-O3"],
                    "nvcc": ["-O3", "--extended-lambda", "--use_fast_math"],
                },
            )
        ],
        cmdclass={"build_ext": BuildExtension},
    )
