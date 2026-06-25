#pragma once
#include "Defs.h"

#include "Api/Default.h"
#include "Framework/IMicrobenchmark.h"

namespace overheads {

std::unique_ptr<IMicrobenchmark<size_t>> cudaMallocTest();

std::unique_ptr<IMicrobenchmark<size_t>> cudaMallocManagedTest();

std::unique_ptr<IMicrobenchmark<size_t>> cudaHostAllocTest();

std::unique_ptr<IMicrobenchmark<size_t>> cudaFreeTest();

std::unique_ptr<IMicrobenchmark<size_t>> cudaFreeHostTest();

std::unique_ptr<IMicrobenchmark<size_t>> cudaMemsetTest();

std::unique_ptr<IMicrobenchmark<size_t>> cudaMemcpyHostToDeviceTest();

std::unique_ptr<IMicrobenchmark<size_t>> cudaMemcpyDeviceToHostTest();

std::unique_ptr<IMicrobenchmark<size_t>> cudaMemcpyDeviceToDeviceTest();

std::unique_ptr<IMicrobenchmark<size_t>> cudaMemcpyAsyncHostToDeviceTest();

std::unique_ptr<IMicrobenchmark<size_t>> cudaMemcpyAsyncDeviceToHostTest();

std::unique_ptr<IMicrobenchmark<size_t>> cudaMemcpyPinnedHostToDeviceTest();

std::unique_ptr<IMicrobenchmark<size_t>> cudaMemcpyPinnedDeviceToHostTest();

std::unique_ptr<IMicrobenchmark<>> cudaEventCreateTest();

std::unique_ptr<IMicrobenchmark<>> cudaEventDestroyTest();

std::unique_ptr<IMicrobenchmark<>> cudaEventRecordTest();

std::unique_ptr<IMicrobenchmark<>> cudaDeviceSynchronizeTest();

std::unique_ptr<IMicrobenchmark<>> cudaDeviceResetTest();


} // namespace overheads
