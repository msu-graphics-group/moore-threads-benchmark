#pragma once
#include "Defs.h"

#include "Api/Default.h"
#include "Framework/IMicrobenchmark.h"

// Measures the throughput of atomicAdd(), atomicExch() and atomicCAS() in global and shared memory
// SHOC has no atomics tests, the sources are named in 'Atomics/Kernels.cu'
namespace atomics {

// How many neighbouring threads share one word: 1 or the whole block
using Contention = size_t;

// The iterations of the main loop in a kernel, a part of the configuration
using Iterations = size_t;

const std::vector<Contention> &SupportedContention();

size_t TotalThreads();

std::function<double(double, Contention, Iterations)> ToCallsPerSecond();

std::unique_ptr<IMicrobenchmark<Contention, Iterations>> globalAtomicAddTest();
std::unique_ptr<IMicrobenchmark<Contention, Iterations>> sharedAtomicAddTest();

std::unique_ptr<IMicrobenchmark<Contention, Iterations>> globalAtomicExchTest();
std::unique_ptr<IMicrobenchmark<Contention, Iterations>> sharedAtomicExchTest();

std::unique_ptr<IMicrobenchmark<Contention, Iterations>> globalAtomicCasTest();
std::unique_ptr<IMicrobenchmark<Contention, Iterations>> sharedAtomicCasTest();

} // namespace atomics
