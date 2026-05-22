#include "Defs.h"

#include "Tests/Overheads.h"

int main() {
  try {
    Api::cudaDeviceProp props;
    std::memset(&props, 0, sizeof(props));
    HANDLE_ERROR(Api::cudaGetDeviceProperties(&props, 0));
    std::cout << props.name << std::endl
              << "     major.minor:         " << props.major << '.' << props.minor << std::endl
              << "     multiProcessorCount: " << props.multiProcessorCount << std::endl
              << "     totalGlobalMem:      " << props.totalGlobalMem << std::endl
              << "     warpSize:            " << props.warpSize << std::endl;
    std::cout << std::endl;

    // Just a draft, will be replaced by a special framework
    auto tests = { overheads::cudaMallocTest(),
                   overheads::cudaFreeTest() };

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
    }
    return 0;
  }
  catch (std::exception &ex) {
    std::cerr << "Error: " << ex.what() << std::endl;
    return 42;
  }
}
