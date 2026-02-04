#!/bin/sh
glslangValidator -V PackXYMega.comp -o PackXYMega.comp.spv -DGLSL -I.. -I/home/frol/PROG/vulkan-musa-benckmark/vk_samples/base02_spheres_pt/../LiteMath -I/home/frol/PROG/kernel_slicer/TINYSTL 
glslangValidator -V CastSingleRayMega.comp -o CastSingleRayMega.comp.spv -DGLSL -I.. -I/home/frol/PROG/vulkan-musa-benckmark/vk_samples/base02_spheres_pt/../LiteMath -I/home/frol/PROG/kernel_slicer/TINYSTL 
glslangValidator -V StupidPathTraceMega.comp -o StupidPathTraceMega.comp.spv -DGLSL -I.. -I/home/frol/PROG/vulkan-musa-benckmark/vk_samples/base02_spheres_pt/../LiteMath -I/home/frol/PROG/kernel_slicer/TINYSTL 
