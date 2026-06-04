#pragma once
#include "Defs.h"

#include "Api/Default.h"
#include "Framework/IMicrobenchmark.h"

namespace overheads {

std::unique_ptr<IMicrobenchmark<size_t>> cudaMallocTest();

std::unique_ptr<IMicrobenchmark<size_t>> cudaFreeTest();

  // TODO:
  // - cudaMallocManaged
  // - cudaMallocHost
  // - cudaFreeHost
  // - cudaMemset
  // - cudaMemcpy
  // - cudaMemcpyAsync
  // - cudaMemcpyPinned
  // - cudaEventCreate
  // - cudaEventDestroy
  // - cudaEventRecord
  // - cudaDeviceSynchronize

} // namespace overheads
