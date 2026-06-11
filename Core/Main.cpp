#include "Defs.h"

#include "Framework/Framework.h"
#include "Tests/Overheads.h"

void PopulateOverheads(Framework &framework) {
  std::vector<size_t> block_sizes = { 1024 * 1024, 4 * 1024 * 1024, 16 * 1024 * 1024 };
  framework.SetTag("Overheads");
  framework.AddLatencyBenchmark(std::move(overheads::cudaMallocTest()),
                                block_sizes, 1, 100, "seconds");
  framework.AddLatencyBenchmark(std::move(overheads::cudaFreeTest()),
                                block_sizes, 1, 100, "seconds");
}

int main() {
  try {
    // Just a draft, will be replaced by a special framework
    Framework framework(0, 0, 0);
    PopulateOverheads(framework);
    framework.SetTextStream(std::cout);
    framework.Run();
    std::cout << std::endl;

    /*auto no_arg_tests = { overheads::cudaEventCreateTest(),
                          overheads::cudaEventDestroyTest(),
                          overheads::cudaEventRecordTest(),
                          overheads::cudaDeviceSynchronizeTest() };

    auto tests = { overheads::cudaMallocTest(),
                   overheads::cudaMallocManagedTest(),
                   overheads::cudaHostAllocTest(),
                   overheads::cudaMemsetTest(),
                   overheads::cudaMemcpyHostToDeviceTest(),
                   overheads::cudaMemcpyDeviceToHostTest(),
                   overheads::cudaMemcpyDeviceToDeviceTest(),
                   overheads::cudaMemcpyAsyncHostToDeviceTest(),
                   overheads::cudaMemcpyAsyncDeviceToHostTest(),
                   overheads::cudaMemcpyPinnedHostToDeviceTest(),
                   overheads::cudaMemcpyPinnedDeviceToHostTest(),
                   overheads::cudaFreeTest(),
                   overheads::cudaFreeHostTest() };

    for(auto &test : no_arg_tests){
      test->Configure(1,100);
      test->Init();
      auto times = test->Run();
      assert(times.size() == 1);
      test->CleanUp();

      std::cout << test->Name() << ": " << std::endl
                << "    " << times.front() << " seconds"
                << std::endl
                << std::endl;
    }

    for (auto &test : tests) {
      test->Configure(1, 100, 1024 * 1024);
      test->Init();
      auto one_mbyte = test->Run();
      assert(one_mbyte.size() == 1);
      test->CleanUp();

      test->Configure(1, 100, 16 * 1024 * 1024);
      test->Init();
      auto sixteen_mbytes = test->Run();
      assert(sixteen_mbytes.size() == 1);
      test->CleanUp();

      std::cout << test->Name() << ":" << std::endl
                << "         1024 * 1024: " << one_mbyte.front() << " seconds" << std::endl
                << "    16 * 1024 * 1024: " << sixteen_mbytes.front() << " seconds" << std::endl;
      std::cout << std::endl;
    }*/

    return 0;
  }
  catch (std::exception &ex) {
    std::cerr << "Error: " << ex.what() << std::endl;
    return 42;
  }
}
