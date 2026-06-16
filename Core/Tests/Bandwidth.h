#pragma once
#include "Defs.h"

#include "Api/Default.h"
#include "Framework/IMicrobenchmark.h"

namespace bandwidth {

std::unique_ptr<IMicrobenchmark<size_t>> cudaRuntimeTest();

std::unique_ptr<IMicrobenchmark<size_t>> cudaMemcpyHostToDeviceTest();

std::unique_ptr<IMicrobenchmark<size_t>> cudaMemcpyDeviceToHostTest();

std::unique_ptr<IMicrobenchmark<size_t>> cudaMemcpyDeviceToDeviceTest();

std::unique_ptr<IMicrobenchmark<size_t>> cudaMemcpyPinnedHostToDeviceTest();

std::unique_ptr<IMicrobenchmark<size_t>> cudaMemcpyPinnedDeviceToHostTest();

std::unique_ptr<IMicrobenchmark<size_t>> cudaMemcpyManagedToDeviceTest();

std::unique_ptr<IMicrobenchmark<size_t>> cudaMemcpyDeviceToManagedTest();

std::unique_ptr<IMicrobenchmark<>> cudaKernelTest();

} // namespace bandwidth
