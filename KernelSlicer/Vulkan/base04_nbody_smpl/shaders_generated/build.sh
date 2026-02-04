#!/bin/sh
glslangValidator -V kernel1D_GenerateBodies.comp -o kernel1D_GenerateBodies.comp.spv -DGLSL -I.. -I/home/frol/PROG/vulkan-musa-benckmark/vk_samples/base04_nbody_smpl/../LiteMath -I/home/frol/PROG/kernel_slicer/TINYSTL 
glslangValidator -V kernel1D_UpdateVelocity.comp -o kernel1D_UpdateVelocity.comp.spv -DGLSL -I.. -I/home/frol/PROG/vulkan-musa-benckmark/vk_samples/base04_nbody_smpl/../LiteMath -I/home/frol/PROG/kernel_slicer/TINYSTL 
glslangValidator -V kernel1D_UpdatePosition.comp -o kernel1D_UpdatePosition.comp.spv -DGLSL -I.. -I/home/frol/PROG/vulkan-musa-benckmark/vk_samples/base04_nbody_smpl/../LiteMath -I/home/frol/PROG/kernel_slicer/TINYSTL 
glslangValidator -V z_memcpy.comp -o z_memcpy.comp.spv
