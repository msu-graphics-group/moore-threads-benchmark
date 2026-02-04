#!/bin/sh
glslangValidator -V kernel2D_EvaluateTextures.comp -o kernel2D_EvaluateTextures.comp.spv -DGLSL -I.. -I/home/frol/PROG/vulkan-musa-benckmark/vk_samples/base08_proctex_v1/../LiteMath -I/home/frol/PROG/kernel_slicer/TINYSTL 
