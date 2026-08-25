#include "Defs.h"

#include "Framework/Framework.h"
#include "Framework/BlockSizeGenerator.h"
#include "Tests/Overheads.h"
#include "Tests/Bandwidth.h"


// Since we need reproducible results, we use the same seed for all random number generators
constexpr uint32_t RNG_SEED = 42;

// Which percent of the memory we can use in a test?
constexpr double MAX_MEMORY_USAGE = 0.2;

// How much memory we can allocate in a test?
constexpr size_t MAX_ALLOCATED_MEMORY = (size_t)1024 * 1024 * 1024;

// How many block sizes we want to generate per each test?
// Some tests may increase or decrease this number
constexpr size_t DEFAULT_NUMBER_OF_BLOCK_SIZES = 40;

// How many iterations we want to run per each combination of test and block size?
// Does not include subiterations
constexpr size_t DEFAULT_NUBER_OF_ITERATIONS = 5;



auto ToSeconds = [](double seconds, auto...) -> double {
  return seconds;
};

std::function<double(double, size_t)> ToBytesPerSecond = [](double seconds, size_t size_in_bytes) {
  return seconds > 1e-9 ? size_in_bytes / seconds : 0.0;
};

std::vector<size_t> SpawnMemoryBlocks(size_t n_blocks, size_t min_block_size, size_t bytes_per_element) {
  size_t divisor = BlockSizeGenerator::WarpSize();
  size_t lo = min_block_size * bytes_per_element;
  size_t hi = BlockSizeGenerator::MaxBlockSize(bytes_per_element, MAX_MEMORY_USAGE, MAX_ALLOCATED_MEMORY);
  std::mt19937 gen{RNG_SEED};

  std::vector<size_t> res;

  // Generate 40% of blocks uniformly
  {
    auto blocks = BlockSizeGenerator::Uniform(lo, hi, divisor, (n_blocks * 4) / 10);
    res.insert(res.end(), blocks.begin(), blocks.end());
  }

  // Generate 40% of blocks exponentially
  {
    size_t n = (n_blocks * 4) / 10;
    auto blocks = BlockSizeGenerator::PowerOfTwo(lo, hi);
    while (blocks.size() > n) {
      auto drop = gen() % blocks.size();
      blocks[drop] = blocks.back();
      blocks.pop_back();
    }
    res.insert(res.end(), blocks.begin(), blocks.end());
  }

  // Generate the last part randomly
  {
    auto blocks = BlockSizeGenerator::Random(lo, hi, divisor, n_blocks - res.size(), RNG_SEED);
    res.insert(res.end(), blocks.begin(), blocks.end());
  }

  std::sort(res.begin(), res.end());
  return res;
}

void PopulateOverheads(Framework &framework) {
  framework.SetTag("Overheads");
  std::function<double(double)> to_seconds_v1 = ToSeconds;
  std::function<double(double, size_t)> to_seconds_v2 = ToSeconds;

  size_t n_iter = DEFAULT_NUBER_OF_ITERATIONS;
  size_t min_block = 1024;
  std::vector<size_t> blocks_x1 = SpawnMemoryBlocks(n_iter, min_block, 1);
  std::vector<size_t> blocks_x2 = SpawnMemoryBlocks(n_iter, min_block, 2);
  std::vector<size_t> blocks_x4 = SpawnMemoryBlocks(n_iter, min_block, 4);

  // Events
  framework.AddBenchmark(std::move(overheads::cudaEventCreateTest()),
                         n_iter * 1000, 64, Unit::Seconds, to_seconds_v1);
  framework.AddBenchmark(std::move(overheads::cudaEventRecordTest()),
                         n_iter * 1000, 64, Unit::Seconds, to_seconds_v1);
  framework.AddBenchmark(std::move(overheads::cudaEventDestroyTest()),
                         n_iter * 1000, 64, Unit::Seconds, to_seconds_v1);

  // Memory allocation
  framework.AddBenchmark(std::move(overheads::cudaMallocTest()), blocks_x4,
                         n_iter, 4, Unit::Seconds, to_seconds_v2, WhoIsBetter::NeedMinMax);
  framework.AddBenchmark(std::move(overheads::cudaMallocManagedTest()), blocks_x4,
                         n_iter, 4, Unit::Seconds, to_seconds_v2, WhoIsBetter::NeedMinMax);
  framework.AddBenchmark(std::move(overheads::cudaFreeTest()), blocks_x4,
                         n_iter, 4, Unit::Seconds, to_seconds_v2, WhoIsBetter::NeedMinMax);
  framework.AddBenchmark(std::move(overheads::cudaHostAllocTest()), blocks_x4,
                         n_iter, 4, Unit::Seconds, to_seconds_v2, WhoIsBetter::NeedMinMax);
  framework.AddBenchmark(std::move(overheads::cudaFreeHostTest()), blocks_x4,
                         n_iter, 4, Unit::Seconds, to_seconds_v2, WhoIsBetter::NeedMinMax);

  // Memory copy
  framework.AddBenchmark(std::move(overheads::cudaMemsetTest()), blocks_x1,
                         n_iter, 50, Unit::Seconds, to_seconds_v2, WhoIsBetter::NeedMinMax);

  framework.AddBenchmark(std::move(overheads::cudaMemcpyHostToDeviceTest()), blocks_x2,
                         n_iter, 5, Unit::Seconds, to_seconds_v2, WhoIsBetter::NeedMinMax);
  framework.AddBenchmark(std::move(overheads::cudaMemcpyDeviceToHostTest()), blocks_x2,
                         n_iter, 5, Unit::Seconds, to_seconds_v2, WhoIsBetter::NeedMinMax);
  framework.AddBenchmark(std::move(overheads::cudaMemcpyDeviceToDeviceTest()), blocks_x2,
                         n_iter, 5, Unit::Seconds, to_seconds_v2, WhoIsBetter::NeedMinMax);

  framework.AddBenchmark(std::move(overheads::cudaMemcpyPinnedHostToDeviceTest()), blocks_x1,
                         n_iter, 1, Unit::Seconds, to_seconds_v2, WhoIsBetter::NeedMinMax);
  framework.AddBenchmark(std::move(overheads::cudaMemcpyPinnedDeviceToHostTest()), blocks_x1,
                         n_iter, 1, Unit::Seconds, to_seconds_v2, WhoIsBetter::NeedMinMax);

  framework.AddBenchmark(std::move(overheads::cudaMemcpyAsyncHostToDeviceTest()), blocks_x1,
                         n_iter, 1, Unit::Seconds, to_seconds_v2, WhoIsBetter::NeedMinMax);
  framework.AddBenchmark(std::move(overheads::cudaMemcpyAsyncDeviceToHostTest()), blocks_x1,
                         n_iter, 1, Unit::Seconds, to_seconds_v2, WhoIsBetter::NeedMinMax);

  // Device
  framework.AddBenchmark(std::move(overheads::cudaDeviceSynchronizeTest()),
                         n_iter * 100, 10, Unit::Seconds, to_seconds_v1);
  framework.AddBenchmark(std::move(overheads::cudaDeviceResetTest()),
                         n_iter * 1000, 1, Unit::Seconds, to_seconds_v1);

}

void PopulateBandwidth(Framework &framework) {
  framework.SetTag("Bandwidth");

  size_t n_iter = DEFAULT_NUBER_OF_ITERATIONS;
  //size_t min_block = 1024;
  size_t min_block = 1024 * 1024;
  std::vector<size_t> blocks_x1 = SpawnMemoryBlocks(n_iter, min_block, 1);
  std::vector<size_t> blocks_x2 = SpawnMemoryBlocks(n_iter, min_block, 2);
  std::vector<size_t> blocks_x4 = SpawnMemoryBlocks(n_iter, min_block, 4);

  framework.AddBenchmark(std::move(bandwidth::cudaMemcpyHostToDeviceTest()),
                         blocks_x2, n_iter, 2, Unit::BytesPerSecond,
                         ToBytesPerSecond, WhoIsBetter::HigherIsBetter);

  framework.AddBenchmark(std::move(bandwidth::cudaMemcpyDeviceToHostTest()),
                         blocks_x2, n_iter, 2, Unit::BytesPerSecond,
                         ToBytesPerSecond, WhoIsBetter::HigherIsBetter);

  framework.AddBenchmark(std::move(bandwidth::cudaMemcpyDeviceToDeviceTest()),
                         blocks_x2, n_iter, 4, Unit::BytesPerSecond,
                         ToBytesPerSecond, WhoIsBetter::HigherIsBetter);

  framework.AddBenchmark(std::move(bandwidth::cudaMemcpyPinnedHostToDeviceTest()),
                         blocks_x1, n_iter, 2, Unit::BytesPerSecond,
                         ToBytesPerSecond, WhoIsBetter::HigherIsBetter);

  framework.AddBenchmark(std::move(bandwidth::cudaMemcpyPinnedDeviceToHostTest()),
                         blocks_x1, n_iter, 2, Unit::BytesPerSecond,
                         ToBytesPerSecond, WhoIsBetter::HigherIsBetter);

  framework.AddBenchmark(std::move(bandwidth::cudaMemcpyManagedToDeviceTest()),
                         blocks_x1, n_iter, 2, Unit::BytesPerSecond,
                         ToBytesPerSecond, WhoIsBetter::HigherIsBetter);

  framework.AddBenchmark(std::move(bandwidth::cudaMemcpyDeviceToManagedTest()),
                         blocks_x1, n_iter, 2, Unit::BytesPerSecond,
                         ToBytesPerSecond, WhoIsBetter::HigherIsBetter);

#if defined(API_CUDA)
  framework.AddBenchmark(std::move(bandwidth::sharedMemoryReadTest()),
                         blocks_x1, n_iter, 10, Unit::BytesPerSecond,
                         ToBytesPerSecond, WhoIsBetter::HigherIsBetter);

  framework.AddBenchmark(std::move(bandwidth::sharedMemoryWriteTest()),
                         blocks_x1, n_iter, 10, Unit::BytesPerSecond,
                         ToBytesPerSecond, WhoIsBetter::HigherIsBetter);

  framework.AddBenchmark(std::move(bandwidth::constantMemoryReadTest()),
                         blocks_x1, n_iter, 10, Unit::BytesPerSecond,
                         ToBytesPerSecond, WhoIsBetter::HigherIsBetter);
#endif
}

int main() {
  try {
    Framework framework;
    framework.ExcludeIterations(0.1,  // 10% for warm-up
                                0.2); // 20% for outliers
    framework.SetTextStream(std::cout);

    PopulateOverheads(framework);
    PopulateBandwidth(framework);
    framework.Run();
    std::cout << std::endl;

    return 0;
  }
  catch (std::exception &ex) {
    std::cerr << "Error: " << ex.what() << std::endl;
    return 42;
  }
}
