#include "Defs.h"

#include "Framework/Framework.h"
#include "Tests/Overheads.h"

void PopulateOverheads(Framework &framework) {
  framework.SetTag("Overheads");
  std::vector<size_t> block_sizes = { 1024 * 1024, 4 * 1024 * 1024, 16 * 1024 * 1024 };

  // Events
  framework.AddBenchmark(std::move(overheads::cudaEventCreateTest()),
                         100, 32, Unit::Seconds, WhoIsBetter::NeedMinMax);
  framework.AddBenchmark(std::move(overheads::cudaEventRecordTest()),
                         100, 32, Unit::Seconds, WhoIsBetter::NeedMinMax);
  framework.AddBenchmark(std::move(overheads::cudaEventDestroyTest()),
                         100, 32, Unit::Seconds, WhoIsBetter::NeedMinMax);

  // Memory allocation
  framework.AddBenchmark(std::move(overheads::cudaMallocTest()),
                         block_sizes, 100, 100, Unit::Seconds, WhoIsBetter::NeedMinMax);
  framework.AddBenchmark(std::move(overheads::cudaMallocManagedTest()),
                         block_sizes, 100, 100, Unit::Seconds, WhoIsBetter::NeedMinMax);
  framework.AddBenchmark(std::move(overheads::cudaFreeTest()),
                         block_sizes, 100, 100, Unit::Seconds, WhoIsBetter::NeedMinMax);
  framework.AddBenchmark(std::move(overheads::cudaHostAllocTest()),
                         block_sizes, 100, 100, Unit::Seconds, WhoIsBetter::NeedMinMax);
  framework.AddBenchmark(std::move(overheads::cudaFreeHostTest()),
                         block_sizes, 100, 100, Unit::Seconds, WhoIsBetter::NeedMinMax);

  // Memory copy
  framework.AddBenchmark(std::move(overheads::cudaMemsetTest()),
                         block_sizes, 100, 100, Unit::Seconds, WhoIsBetter::NeedMinMax);

  framework.AddBenchmark(std::move(overheads::cudaMemcpyHostToDeviceTest()),
                         block_sizes, 100, 100, Unit::Seconds, WhoIsBetter::NeedMinMax);
  framework.AddBenchmark(std::move(overheads::cudaMemcpyDeviceToHostTest()),
                         block_sizes, 100, 100, Unit::Seconds, WhoIsBetter::NeedMinMax);
  framework.AddBenchmark(std::move(overheads::cudaMemcpyDeviceToDeviceTest()),
                         block_sizes, 100, 100, Unit::Seconds, WhoIsBetter::NeedMinMax);

  framework.AddBenchmark(std::move(overheads::cudaMemcpyPinnedHostToDeviceTest()),
                         block_sizes, 100, 100, Unit::Seconds, WhoIsBetter::NeedMinMax);
  framework.AddBenchmark(std::move(overheads::cudaMemcpyPinnedDeviceToHostTest()),
                         block_sizes, 100, 100, Unit::Seconds, WhoIsBetter::NeedMinMax);

  framework.AddBenchmark(std::move(overheads::cudaMemcpyAsyncHostToDeviceTest()),
                         block_sizes, 100, 100, Unit::Seconds, WhoIsBetter::NeedMinMax);
  framework.AddBenchmark(std::move(overheads::cudaMemcpyAsyncDeviceToHostTest()),
                         block_sizes, 100, 100, Unit::Seconds, WhoIsBetter::NeedMinMax);

  // Device
  framework.AddBenchmark(std::move(overheads::cudaDeviceSynchronizeTest()),
                         100, 100, Unit::Seconds, WhoIsBetter::NeedMinMax);
}

int main() {
  try {
    // Just a draft, will be rewritten
    Framework framework(0, 0, 0);
    PopulateOverheads(framework);
    framework.SetTextStream(std::cout);
    framework.Run();
    std::cout << std::endl;

    return 0;
  }
  catch (std::exception &ex) {
    std::cerr << "Error: " << ex.what() << std::endl;
    return 42;
  }
}
