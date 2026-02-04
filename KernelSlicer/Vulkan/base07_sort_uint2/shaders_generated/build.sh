#!/bin/sh
glslangValidator -V kernel1D_CopyData.comp -o kernel1D_CopyData.comp.spv -DGLSL -I.. -I/home/frol/PROG/vulkan-musa-benckmark/vk_samples/base07_sort_uint2/../LiteMath -I/home/frol/PROG/kernel_slicer/TINYSTL 
glslangValidator -V z_bitonic_uvec2_pass.comp -o z_bitonic_uvec2_pass.comp.spv
glslangValidator -V z_bitonic_uvec2_512.comp  -o z_bitonic_uvec2_512.comp.spv
glslangValidator -V z_bitonic_uvec2_1024.comp -o z_bitonic_uvec2_1024.comp.spv
glslangValidator -V z_bitonic_uvec2_2048.comp -o z_bitonic_uvec2_2048.comp.spv
