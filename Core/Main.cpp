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
    
    return 0;
  }
  catch (std::exception &ex) {
    std::cerr << "Error: " << ex.what() << std::endl;
    return 42;
  }
}
