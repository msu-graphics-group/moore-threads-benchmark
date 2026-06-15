#pragma once
#include "Defs.h"

#include "Api/Default.h"
#include "Framework/IMicrobenchmark.h"

namespace bandwidth {

std::unique_ptr<IMicrobenchmark<size_t>> cudaRuntimeTest();

std::unique_ptr<IMicrobenchmark<>> cudaKernelTest();

} // namespace bandwidth
