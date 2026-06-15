#include "Tests/Bandwidth.h"

#include "Framework/CudaEventBenchmark.h"


//-----------------------
//--- cudaRuntimeTest ---
//-----------------------

namespace {

class cudaKernelImpl : public CudaEventBenchmark<> {
  public:
    cudaKernelImpl() = default;

    virtual std::string Name() const override { return "bandwidth::cudaKernel()"; }

    virtual void SingleRun() override {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
};

} // unnamed namespace


namespace bandwidth {

std::unique_ptr<IMicrobenchmark<>> cudaKernelTest() {
  return std::make_unique<cudaKernelImpl>();
}

}
