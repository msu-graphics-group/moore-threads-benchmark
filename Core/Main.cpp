#include "Defs.h"

#include "Framework/Framework.h"
#include "Tests/Overheads.h"
#include "Tests/Bandwidth.h"


auto ToSeconds = [](double seconds, auto...) -> double {
  return seconds;
};

std::function<double(double, size_t)> ToBytesPerSecond = [](double seconds, size_t size_in_bytes) {
  return seconds > 1e-9 ? size_in_bytes / seconds : 0.0;
};

void PopulateOverheads(Framework &framework) {
  framework.SetTag("Overheads");
  std::function<double(double)> to_seconds_v1 = ToSeconds;
  std::function<double(double, size_t)> to_seconds_v2 = ToSeconds;

  std::vector<size_t> block_sizes = { 1024 * 1024, 4 * 1024 * 1024, 16 * 1024 * 1024 };

  // Events
  framework.AddBenchmark(std::move(overheads::cudaEventCreateTest()),
                         100, 32, Unit::Seconds, to_seconds_v1, WhoIsBetter::NeedMinMax);
  framework.AddBenchmark(std::move(overheads::cudaEventRecordTest()),
                         100, 32, Unit::Seconds, to_seconds_v1, WhoIsBetter::NeedMinMax);
  framework.AddBenchmark(std::move(overheads::cudaEventDestroyTest()),
                         100, 32, Unit::Seconds, to_seconds_v1, WhoIsBetter::NeedMinMax);

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
                         100, 100, Unit::Seconds, to_seconds_v1, WhoIsBetter::NeedMinMax);
}

void PopulateBandwidth(Framework &framework) {
  framework.SetTag("Bandwidth");
  std::vector<size_t> block_sizes = { 1024 * 1024 };

  framework.AddBenchmark(std::move(bandwidth::cudaRuntimeTest()),
                         block_sizes, 100, 100, Unit::BytesPerSecond,
                         ToBytesPerSecond, WhoIsBetter::HigherIsBetter);

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
                         100, 100, Unit::Seconds, ToSeconds, WhoIsBetter::HigherIsBetter);
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
