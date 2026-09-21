`moore-threads-benchmark` is an unofficial collection of performance tests for Moore Threads GPUs.

### Key Features
* Independent performance comparison of MTT GPUs against NVIDIA and AMD.
* No MUSA-specific optimizations. The exact same code is used for MUSA, CUDA, and HIP (or Vulkan for all).
* We adapt popular third-party tests rather than developing our own.

### Included Benchmarks
Currently, the repository contains the following test suites:

* **Core** *(MUSA, CUDA, HIP)*: A set of microbenchmarks inspired by [SHOC](https://github.com/vetter/shoc). It runs synthetic kernels to measure achievable peak `FLOPS` and bandwidth (`B/s`). Since the original [SHOC](https://github.com/vetter/shoc) project is obsolete, we have prepared a new framework for these tests.

* **[KernelSlicer](https://github.com/Ray-Tracing-Systems/kernel_slicer)** *(Vulkan, HIP, CUDA, MUSA)*: A source-to-source compiler for generating GPGPU code from annotated C++20 classes. It features built-in benchmarking tools that we use to estimate performance across different computing patterns.

* **[CUDAMicroBench](https://github.com/passlab/CUDAMicroBench)** *(CUDA, MUSA)*: A research project designed to test optimization techniques and evaluate peak performance on modern NVIDIA GPUs. Originally developed with newer architectures like NVIDIA Ampere in mind (unlike older suites such as [Rodinia](https://rodinia.cs.virginia.edu/)), it provides more relevant benchmarks for contemporary hardware.
