"""
Setup and installation script for Elastic-MTP.
Supports AOT C++/CUDA extension compilation and editable installation via `pip install -e .`.
"""
import os
import sys
from setuptools import setup, find_packages

def get_extensions_and_cmdclass():
    ext_modules = []
    cmdclass = {}
    try:
        import torch
        from torch.utils.cpp_extension import CUDAExtension, CppExtension, BuildExtension
        cmdclass["build_ext"] = BuildExtension
        
        cxx_args = ["/std:c++17"] if sys.platform == "win32" else ["-std=c++17"]
        
        cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
        if torch.cuda.is_available() and cuda_home:
            ext = CUDAExtension(
                name="elastic_mtp._C",
                sources=[
                    "csrc/fused_entropy.cpp",
                    "csrc/fused_entropy_kernel.cu",
                ],
                extra_compile_args={
                    "cxx": cxx_args + ["-DWITH_CUDA"],
                    "nvcc": ["-O3", "--use_fast_math", "-Xcompiler", "/std:c++17", "-DWITH_CUDA"] if sys.platform == "win32" else ["-O3", "--use_fast_math", "-std=c++17", "-DWITH_CUDA"]
                }
            )
            ext_modules.append(ext)
        else:
            ext = CppExtension(
                name="elastic_mtp._C",
                sources=["csrc/fused_entropy.cpp"],
                extra_compile_args=cxx_args
            )
            ext_modules.append(ext)
    except Exception as e:
        print(f"[elastic-mtp setup] Building pure Python package: {e}")
        
    return ext_modules, cmdclass

ext_modules, cmdclass = get_extensions_and_cmdclass()

setup(
    name="elastic-mtp",
    version="2.0.0",
    description="Uncertainty-Aware Dynamic Horizon Multi-Token Prediction Engine",
    author="Elastic-MTP Deepmind Engineering Team",
    packages=find_packages(include=["elastic_mtp", "elastic_mtp.*"]),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.22.0",
        "transformers>=4.38.0"
    ],
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
