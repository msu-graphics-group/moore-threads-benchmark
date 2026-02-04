#!/bin/sh
glslangValidator -V kernel1D_ArraySumm.comp -o kernel1D_ArraySumm.comp.spv -DGLSL -I.. -I/home/frol/PROG/vulkan-musa-benckmark/vk_samples/base03_reduction1/../LiteMath -I/home/frol/PROG/kernel_slicer/TINYSTL 
glslangValidator -V kernel1D_ArraySumm_Reduction.comp -o kernel1D_ArraySumm_Reduction.comp.spv -DGLSL -I.. -I/home/frol/PROG/vulkan-musa-benckmark/vk_samples/base03_reduction1/../LiteMath -I/home/frol/PROG/kernel_slicer/TINYSTL 
