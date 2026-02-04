#!/bin/sh
glslangValidator -V kernel1D_Test.comp -o kernel1D_Test.comp.spv -DGLSL -I.. -I/home/frol/PROG/vulkan-musa-benckmark/vk_samples/base05_append_buf/../LiteMath -I/home/frol/PROG/kernel_slicer/TINYSTL 
glslangValidator -V z_scan_int_block.comp     -o z_scan_int_block.comp.spv
glslangValidator -V z_scan_int_propagate.comp -o z_scan_int_propagate.comp.spv
