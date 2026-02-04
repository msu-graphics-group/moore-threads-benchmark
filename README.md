`moore-threads-benchmark` is an unofficial collection of third-party performance tests ported to Moore Threads GPUs. The goal of this repository is to provide a realistic comparison of their performance with different GPU platforms, without applying MUSA-specific optimizations. This approach reflects a common developer scenario: running existing software rather than developing from scratch.

Currently, `moore-threads-benchmark` includes:

[KernelSlicer](https://github.com/Ray-Tracing-Systems/kernel_slicer) (Vulkan, HiP, CUDA, MUSA)
A source-to-source compiler for generating GPGPU code from annotated C++20 classes. It has a built-in benchmarking tools that helps us compare Moore Threads GPUs with various competing platforms from AMD, Intel and NVIDIA.

[CUDAMicroBench](https://github.com/passlab/CUDAMicroBench) (CUDA, MUSA)
A research project designed to test CUDA optimization techniques and evaluate peak performance on modern NVIDIA GPUs. In oppose to older benchmark suites ([SHOC](https://github.com/vetter/shoc), [Rodinia](https://github.com/yuhc/gpu-rodinia)), CUDAMicroBench is developed with awareness of relatively modern architectures such as NVIDIA Ampere.
