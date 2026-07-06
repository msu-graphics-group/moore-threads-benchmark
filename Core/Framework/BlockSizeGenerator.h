#pragma once
#include "Defs.h"

#include "Api/Default.h"


// A static class that generates block sizes for benchmarking purposes
class BlockSizeGenerator {
  public:
    BlockSizeGenerator() = delete;

    // Returns the maximum block size that can be allocated. The allocated memory must be:
    // - no greater than 'max_allocated_memory'
    // - no greater than 'max_memory_usage' * 'totalGlobalMem'
    static size_t MaxBlockSize(size_t bytes_per_element, double max_memory_usage, size_t max_allocated_memory);

    // Use this as 'divisor' to utilize all threads in a warp
    static size_t WarpSize() { return Props().warpSize; }

    // Creates 'n_blocks' uniformly distributed block sizes within the range [lo, hi]
    // Each one is a multiple of 'divisor'
    // Example: <1, 6, 2, 3> -> { 2, 4, 6 }
    static std::vector<size_t> Uniform(size_t lo, size_t hi, size_t divisor, size_t n_blocks);

    // Similar to 'Uniform', but the block sizes are exponentially distributed with the given 'base'
    // After generation, each record will be updated to become multiples of 'divisor'
    // Example: <1000, 5000, 100, 2> -> { 1000, 2000, 4000 }
    static std::vector<size_t> Exponential(size_t lo, size_t hi, size_t divisor, double base);

    // Syntax sugar for 'Exponential()' with base 2
    static std::vector<size_t> PowerOfTwo(size_t lo, size_t hi) {
      return Exponential(lo, hi, 1, 2.0);
    }

    // Similar to 'Uniform', but the block sizes are randomly distributed within the range [lo, hi]
    // After generation, each record will be updated to become multiples of 'divisor'
    // Example: <100, 200, 5, 3, 42> -> { 135, 175, 105 }
    static std::vector<size_t> Random(size_t lo, size_t hi, size_t divisor, size_t n_blocks, uint64_t seed);

  private:
    static Api::cudaDeviceProp &Props();
};
