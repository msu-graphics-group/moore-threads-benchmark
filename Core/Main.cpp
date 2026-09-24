#include "Defs.h"

#include "Framework/Framework.h"
#include "Framework/BlockSizeGenerator.h"
#include "Tests/Overheads.h"
#include "Tests/Bandwidth.h"
#include "Tests/Compute.h"


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
                         n_iter * 4, 1, Unit::Seconds, to_seconds_v1);

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

#if defined(API_CUDA)

// A draft of the calibration strategy, it lives outside the micro-tests

// How long a single run of a calibrated test should take, in seconds?
constexpr double CALIBRATION_SECONDS = 0.02;

// How many iterations a calibrated test may use?
constexpr size_t CALIBRATION_MIN_ITERATIONS = 16;
constexpr size_t CALIBRATION_MAX_ITERATIONS = 65536;

// Runs a short probe and scales the iterations (the last argument) to fit 'CALIBRATION_SECONDS'
// It knows nothing about the test, only the public interface of 'IMicrobenchmark' is used
template <typename T>
size_t CalibrateIterations(std::unique_ptr<IMicrobenchmark<T, size_t>> &&benchmark, T probe_arg) {
  constexpr size_t probe_iterations = 32;

  // Two outer iterations, the first one is a warm-up
  benchmark->Configure(2, 1, probe_arg, probe_iterations);
  double seconds = benchmark->Run().back();
  if (seconds < 1e-9) {
    return CALIBRATION_MAX_ITERATIONS;
  }

  auto scaled = static_cast<size_t>(probe_iterations * CALIBRATION_SECONDS / seconds);
  return std::clamp(scaled, CALIBRATION_MIN_ITERATIONS, CALIBRATION_MAX_ITERATIONS);
}

// A compute test is configured by the level of ILP and by the iterations of its kernel
using ComputeTest = std::unique_ptr<IMicrobenchmark<compute::Ilp, compute::Iterations>>;

// Calibrates a single test on its own and adds it with one configuration per level of ILP
void AddComputeTest(Framework &framework, ComputeTest (*make_test)(), size_t n_iter, size_t n_sub_iter,
                    Unit unit, size_t ops_per_step) {
  size_t iterations = CalibrateIterations(make_test(), compute::Ilp{ 2 });

  std::vector<std::tuple<compute::Ilp, compute::Iterations>> configs;
  for (compute::Ilp ilp : compute::SupportedIlp()) {
    configs.emplace_back(ilp, iterations);
  }

  framework.AddBenchmark(make_test(), configs, n_iter, n_sub_iter, unit,
                         compute::ToCallsPerSecond(ops_per_step), WhoIsBetter::HigherIsBetter);
}

// Packed types process several values per operation
template <typename T>
size_t Lanes() {
  return std::is_same_v<T, compute::half2_t> ? 2 : 1;
}

// Adds sin(), cos(), exp() and log() for the given type, unless it has no intrinsics for them
template <typename T>
void AddMathTests(Framework &framework, size_t n_iter, size_t n_sub_iter) {
  if (!compute::IsMathSupported<T>()) {
    return;
  }

  // The framework has no unit for 'function calls per second', so they are reported
  // as Flops: one 'flop' here means one evaluation of the tested function
  size_t calls = compute::CALLS_PER_STEP * Lanes<T>();
  AddComputeTest(framework, compute::sinTest<T>, n_iter, n_sub_iter, Unit::Flops, calls);
  AddComputeTest(framework, compute::cosTest<T>, n_iter, n_sub_iter, Unit::Flops, calls);
  AddComputeTest(framework, compute::expTest<T>, n_iter, n_sub_iter, Unit::Flops, calls);
  AddComputeTest(framework, compute::logTest<T>, n_iter, n_sub_iter, Unit::Flops, calls);
}

// Adds add, mul and madd for the given type
template <typename T>
void AddArithmeticTests(Framework &framework, size_t n_iter, size_t n_sub_iter, Unit unit) {
  // Every integer step also performs one XOR, so it is counted too
  size_t mix = std::is_integral_v<T> ? 1 : 0;
  AddComputeTest(framework, compute::addTest<T>,  n_iter, n_sub_iter, unit, (1 + mix) * Lanes<T>());
  AddComputeTest(framework, compute::mulTest<T>,  n_iter, n_sub_iter, unit, (1 + mix) * Lanes<T>());
  AddComputeTest(framework, compute::maddTest<T>, n_iter, n_sub_iter, unit, (2 + mix) * Lanes<T>());
}

void PopulateCompute(Framework &framework) {
  framework.SetTag("Compute");

  size_t n_iter = DEFAULT_NUBER_OF_ITERATIONS;

  // These kernels are long enough, so we do not need many subiterations to hide the overheads
  size_t n_sub_iter = 3;

  AddMathTests<double>(framework, n_iter, n_sub_iter);
  AddArithmeticTests<double>(framework, n_iter, n_sub_iter, Unit::Flops);

  AddMathTests<float>(framework, n_iter, n_sub_iter);
  AddArithmeticTests<float>(framework, n_iter, n_sub_iter, Unit::Flops);

  // Not every toolkit supports float16
  if (compute::IsFp16Supported()) {
    AddMathTests<compute::half_t>(framework, n_iter, n_sub_iter);
    AddArithmeticTests<compute::half_t>(framework, n_iter, n_sub_iter, Unit::Flops);
    AddMathTests<compute::half2_t>(framework, n_iter, n_sub_iter);
    AddArithmeticTests<compute::half2_t>(framework, n_iter, n_sub_iter, Unit::Flops);
  }

  AddArithmeticTests<int32_t>(framework, n_iter, n_sub_iter, Unit::IntOps);
  AddArithmeticTests<uint32_t>(framework, n_iter, n_sub_iter, Unit::IntOps);
}

#endif // API_CUDA

int main() {
  try {
    Framework framework;
    framework.ExcludeIterations(0.1,  // 10% for warm-up
                                0.2); // 20% for outliers
    framework.SetTextStream(std::cout);

    PopulateOverheads(framework);
    PopulateBandwidth(framework);
#if defined(API_CUDA)
    PopulateCompute(framework);
#endif
    framework.Run();
    std::cout << std::endl;

    return 0;
  }
  catch (std::exception &ex) {
    std::cerr << "Error: " << ex.what() << std::endl;
    return 42;
  }
}
