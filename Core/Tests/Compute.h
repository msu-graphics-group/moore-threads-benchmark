#pragma once
#include "Defs.h"

#include "Api/Default.h"
#include "Framework/IMicrobenchmark.h"

// Measures the throughput of sin(), cos(), exp(), log() and of the plain arithmetic
// The methodology is taken from SHOC, 'MaxFlops', and the three arithmetic tests follow its tests
// (src/cuda/level0/MaxFlops.cu, 1608-1906)
//
// By default sin(), cos(), exp() and log() measure the special function unit through the float32
// intrinsics, the library versions are out of scope and stay behind a switch in 'Compute/Kernels.cu'
namespace compute {

// Tags for float16, these types cannot be named in the host code
struct half_t;
struct half2_t;

// The number of independent chains of operations executed by a single thread
using Ilp = size_t;

// The iterations of the main loop in a kernel, a part of the configuration
using Iterations = size_t;

// One step of a transcendental test is a single evaluation of the tested function
// The addition (a subtraction for exp()) that keeps the chain bounded is not counted, so the result
// is lower than the speed of the function alone, e.g. by roughly a quarter for sin() on GTX 1080 Ti
constexpr size_t CALLS_PER_STEP = 1;

// The supported levels of instruction level parallelism, i.e. chains per thread
const std::vector<Ilp> &SupportedIlp();

// Only the toolkit decides, every supported GPU has native float16
bool IsFp16Supported();

// Whether sin(), cos(), exp() and log() can be measured for the given type
// Only float32 has intrinsics for them, so with the special function unit the other types are skipped
template <typename T> bool IsMathSupported();

// The total number of threads used by the kernels, i.e. 'blocks * threads_per_block'
size_t TotalThreads();

// Builds a converter from 'seconds' into 'operations per second'
// An operation is a function evaluation or an arithmetic operation, 'calls_per_step' counts them in one step
std::function<double(double, Ilp, Iterations)> ToCallsPerSecond(size_t calls_per_step);

// s = sin(s) + v1
template <typename T> std::unique_ptr<IMicrobenchmark<Ilp, Iterations>> sinTest();

// s = cos(s) + v1
template <typename T> std::unique_ptr<IMicrobenchmark<Ilp, Iterations>> cosTest();

// s = exp(v1 - s)
template <typename T> std::unique_ptr<IMicrobenchmark<Ilp, Iterations>> expTest();

// s = log(s) + v1
template <typename T> std::unique_ptr<IMicrobenchmark<Ilp, Iterations>> logTest();

// The three tests below also accept 'int32_t' and 'uint32_t', where every step additionally
// performs 's ^= v1', so an integer step has one more operation than a float one

// s = v1 - s, one flop per step
template <typename T> std::unique_ptr<IMicrobenchmark<Ilp, Iterations>> addTest();

// s = s * v1, one flop per step
template <typename T> std::unique_ptr<IMicrobenchmark<Ilp, Iterations>> mulTest();

// s = s * v1 + v1, two flops per step
template <typename T> std::unique_ptr<IMicrobenchmark<Ilp, Iterations>> maddTest();

} // namespace compute
