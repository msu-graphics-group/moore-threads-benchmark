#pragma once
#include "Defs.h"

#include "Api/Default.h"
#include "Microbenchmarks/IMicrobenchmark.h"

namespace overheads {

std::unique_ptr<IMicrobenchmark<size_t>> cudaMallocTest();

std::unique_ptr<IMicrobenchmark<size_t>> cudaFreeTest();

std::unique_ptr<IMicrobenchmark<size_t>> cudaMemcpyHostToDeviceTest();

std::unique_ptr<IMicrobenchmark<size_t>> cudaMemcpyDeviceToHostTest();

std::unique_ptr<IMicrobenchmark<size_t>> cudaMemcpyDeviceToDeviceTest();

std::unique_ptr<IMicrobenchmark<>> cudaEventCreateTest();

std::unique_ptr<IMicrobenchmark<>> cudaEventDestroyTest();

std::unique_ptr<IMicrobenchmark<>> cudaEventRecordTest();

std::unique_ptr<IMicrobenchmark<>> cudaDeviceSynchronizeTest();

  // TODO:
  // - cudaMallocManaged
  // - cudaMallocHost
  // - cudaFreeHost
  // - cudaMemset
  // - cudaMemcpy                   +
  //    - cudaMemcpyHostToDevice
  //    - cudaMemcpyDeviceToHost
  //    - cudaMemcpyDeviceToDevice
  // - cudaMemcpyAsync
  // - cudaMemcpyPinned
  // - cudaEventCreate              +
  // - cudaEventDestroy             +
  // - cudaEventRecord              +
  // - cudaDeviceSynchronize        +

} // namespace overheads
