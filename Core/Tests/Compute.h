#pragma once
#include "Defs.h"

#include "Api/Default.h"
#include "Framework/IMicrobenchmark.h"

// Measures the throughput of sin(), cos(), exp(), log() and of the plain arithmetic
// The methodology and the three arithmetic tests are taken from SHOC, 'MaxFlops'
// (src/cuda/level0/MaxFlops.cu, 1608-1906)
//
// We call the regular 'sinf()' and not the '__sinf()' intrinsics: we need the speed that
// an existing application gets, not the raw speed of the special function unit
namespace compute {

// Tags for float16, these types cannot be named in the host code
struct half_t;
struct half2_t;

// The number of independent chains of operations executed by a single thread
using Ilp = size_t;

// One step of a transcendental test is a single evaluation of the tested function
// The addition that keeps the chain bounded is not counted, it is much cheaper
constexpr size_t CALLS_PER_STEP = 1;

// The supported levels of instruction level parallelism, i.e. chains per thread
const std::vector<Ilp> &SupportedIlp();

// Neither every toolkit nor every GPU can evaluate these functions in float16
bool IsFp16Supported();

// The total number of threads used by the kernels, i.e. 'blocks * threads_per_block'
size_t TotalThreads();

// Runs a probe kernel with sin(), the most expensive function, and scales the iterations
template <typename T> size_t Calibrate();

// Builds a converter from 'seconds' into 'function evaluations per second'
std::function<double(double, Ilp)> ToCallsPerSecond(size_t calls_per_step, size_t iterations);

// s = sin(s) + v1
template <typename T> std::unique_ptr<IMicrobenchmark<Ilp>> sinTest(size_t iterations);

// s = cos(s) + v1
template <typename T> std::unique_ptr<IMicrobenchmark<Ilp>> cosTest(size_t iterations);

// s = exp(v1 - s)
template <typename T> std::unique_ptr<IMicrobenchmark<Ilp>> expTest(size_t iterations);

// s = log(s) + v1
template <typename T> std::unique_ptr<IMicrobenchmark<Ilp>> logTest(size_t iterations);

// s = v1 - s, one real flop per step
template <typename T> std::unique_ptr<IMicrobenchmark<Ilp>> addTest(size_t iterations);

// s = s * v1, one real flop per step
template <typename T> std::unique_ptr<IMicrobenchmark<Ilp>> mulTest(size_t iterations);

// s = s * v1 + v1, two real flops per step
template <typename T> std::unique_ptr<IMicrobenchmark<Ilp>> maddTest(size_t iterations);

} // namespace compute
