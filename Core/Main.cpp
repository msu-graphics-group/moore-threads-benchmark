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
constexpr size_t DEFAULT_NUMBER_OF_BLOCK_SIZES = 100;



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

  // Generate 1/3 of blocks uniformly
  {
    auto blocks = BlockSizeGenerator::Uniform(lo, hi, divisor, n_blocks / 3);
    res.insert(res.end(), blocks.begin(), blocks.end());
  }

  // Generate 1/3 of blocks exponentially
  {
    size_t n = n_blocks / 3;
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

  std::vector<size_t> single_blocks = SpawnMemoryBlocks(DEFAULT_NUMBER_OF_BLOCK_SIZES, 1024, 1);
  std::vector<size_t> double_blocks = SpawnMemoryBlocks(DEFAULT_NUMBER_OF_BLOCK_SIZES, 1024, 2);
  std::vector<size_t> block_sizes = { 1024 * 1024 };

  // Events
  framework.AddBenchmark(std::move(overheads::cudaEventCreateTest()),
                         100, 32, Unit::Seconds, to_seconds_v1);
  framework.AddBenchmark(std::move(overheads::cudaEventRecordTest()),
                         100, 32, Unit::Seconds, to_seconds_v1);
  framework.AddBenchmark(std::move(overheads::cudaEventDestroyTest()),
                         100, 32, Unit::Seconds, to_seconds_v1);

  // Memory allocation
  framework.AddBenchmark(std::move(overheads::cudaMallocTest()), block_sizes,
                         100, 100, Unit::Seconds, to_seconds_v2, WhoIsBetter::NeedMinMax);
  framework.AddBenchmark(std::move(overheads::cudaMallocManagedTest()), block_sizes,
                         100, 100, Unit::Seconds, to_seconds_v2, WhoIsBetter::NeedMinMax);
  framework.AddBenchmark(std::move(overheads::cudaFreeTest()), block_sizes,
                         100, 100, Unit::Seconds, to_seconds_v2, WhoIsBetter::NeedMinMax);
  framework.AddBenchmark(std::move(overheads::cudaHostAllocTest()), block_sizes,
                         100, 100, Unit::Seconds, to_seconds_v2, WhoIsBetter::NeedMinMax);
  framework.AddBenchmark(std::move(overheads::cudaFreeHostTest()), block_sizes,
                         100, 100, Unit::Seconds, to_seconds_v2, WhoIsBetter::NeedMinMax);

  // Memory copy
  framework.AddBenchmark(std::move(overheads::cudaMemsetTest()), block_sizes,
                         100, 100, Unit::Seconds, to_seconds_v2, WhoIsBetter::NeedMinMax);

  framework.AddBenchmark(std::move(overheads::cudaMemcpyHostToDeviceTest()), block_sizes,
                         100, 100, Unit::Seconds, to_seconds_v2, WhoIsBetter::NeedMinMax);
  framework.AddBenchmark(std::move(overheads::cudaMemcpyDeviceToHostTest()), block_sizes,
                         100, 100, Unit::Seconds, to_seconds_v2, WhoIsBetter::NeedMinMax);
  framework.AddBenchmark(std::move(overheads::cudaMemcpyDeviceToDeviceTest()), block_sizes,
                         100, 100, Unit::Seconds, to_seconds_v2, WhoIsBetter::NeedMinMax);

  framework.AddBenchmark(std::move(overheads::cudaMemcpyPinnedHostToDeviceTest()), block_sizes,
                         100, 100, Unit::Seconds, to_seconds_v2, WhoIsBetter::NeedMinMax);
  framework.AddBenchmark(std::move(overheads::cudaMemcpyPinnedDeviceToHostTest()), block_sizes,
                         100, 100, Unit::Seconds, to_seconds_v2, WhoIsBetter::NeedMinMax);

  framework.AddBenchmark(std::move(overheads::cudaMemcpyAsyncHostToDeviceTest()), block_sizes,
                         100, 100, Unit::Seconds, to_seconds_v2, WhoIsBetter::NeedMinMax);
  framework.AddBenchmark(std::move(overheads::cudaMemcpyAsyncDeviceToHostTest()), block_sizes,
                         100, 100, Unit::Seconds, to_seconds_v2, WhoIsBetter::NeedMinMax);

  // Device
  framework.AddBenchmark(std::move(overheads::cudaDeviceSynchronizeTest()),
                         100, 100, Unit::Seconds, to_seconds_v1);
  framework.AddBenchmark(std::move(overheads::cudaDeviceResetTest()),
                         100, 1, Unit::Seconds, to_seconds_v1);
                         
}

void PopulateBandwidth(Framework &framework) {
  framework.SetTag("Bandwidth");
  std::vector<size_t> block_sizes = { 1024 * 1024 };

  framework.AddBenchmark(std::move(bandwidth::cudaMemcpyHostToDeviceTest()),
                         block_sizes, 100, 100, Unit::BytesPerSecond,
                         ToBytesPerSecond, WhoIsBetter::HigherIsBetter);

  framework.AddBenchmark(std::move(bandwidth::cudaMemcpyDeviceToHostTest()),
                         block_sizes, 100, 100, Unit::BytesPerSecond,
                         ToBytesPerSecond, WhoIsBetter::HigherIsBetter);

  framework.AddBenchmark(std::move(bandwidth::cudaMemcpyDeviceToDeviceTest()),
                         block_sizes, 100, 100, Unit::BytesPerSecond,
                         ToBytesPerSecond, WhoIsBetter::HigherIsBetter);

  framework.AddBenchmark(std::move(bandwidth::cudaMemcpyPinnedHostToDeviceTest()),
                         block_sizes, 100, 100, Unit::BytesPerSecond,
                         ToBytesPerSecond, WhoIsBetter::HigherIsBetter);

  framework.AddBenchmark(std::move(bandwidth::cudaMemcpyPinnedDeviceToHostTest()),
                         block_sizes, 100, 100, Unit::BytesPerSecond,
                         ToBytesPerSecond, WhoIsBetter::HigherIsBetter);

  framework.AddBenchmark(std::move(bandwidth::cudaMemcpyManagedToDeviceTest()),
                         block_sizes, 100, 100, Unit::BytesPerSecond,
                         ToBytesPerSecond, WhoIsBetter::HigherIsBetter);

  framework.AddBenchmark(std::move(bandwidth::cudaMemcpyDeviceToManagedTest()),
                         block_sizes, 100, 100, Unit::BytesPerSecond,
                         ToBytesPerSecond, WhoIsBetter::HigherIsBetter);

#if defined(API_CUDA)
  framework.AddBenchmark(std::move(bandwidth::cudaKernelTest()),
                         100, 100, Unit::Seconds, ToSeconds);
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
