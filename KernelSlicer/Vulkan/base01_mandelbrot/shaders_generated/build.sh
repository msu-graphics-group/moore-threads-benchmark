#!/bin/sh
glslangValidator -V kernel2D_process.comp -o kernel2D_process.comp.spv -DGLSL -I.. -I/home/frol/PROG/kernel_slicer/TINYSTL -I/home/frol/PROG/kernel_slicer/apps/LiteMath -I/home/frol/PROG/kernel_slicer/apps/LiteMathAux 
