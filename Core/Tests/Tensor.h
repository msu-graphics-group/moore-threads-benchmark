#pragma once
#include "Framework/IMicrobenchmark.h"

// Measures dense WMMA throughput with FP16 inputs, FP32 accumulation and 16 x 16 x 16 tiles
namespace tensor {

// The number of independent accumulator chains executed by a warp
using Ilp = size_t;

// The iterations of the main loop in a kernel, a part of the configuration
using Iterations = size_t;

// Whether the benchmark name includes its ILP, warp and iteration counts
enum class OutputMode {
  Compact,
  Detailed
};

// An empty string means that both the device and the compiled kernel support this test
// Unsupported backends and devices are skipped, there is no scalar fallback
std::string UnsupportedReason();

// The total number of warps used by the kernel, i.e. 'blocks * warps_per_block'
size_t TotalWarps();

// Builds the WMMA test with the requested form of its name
std::unique_ptr<IMicrobenchmark<Ilp, Iterations>> MakeTest(OutputMode mode = OutputMode::Compact);

// Builds a converter from 'seconds' into 'floating-point operations per second'
// Each warp performs 32 matrix multiply-accumulate calls per iteration, regardless of ILP
std::function<double(double, Ilp, Iterations)> ToFlops();

} // namespace tensor
