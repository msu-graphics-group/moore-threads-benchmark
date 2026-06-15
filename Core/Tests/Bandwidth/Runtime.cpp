#include "Tests/Bandwidth.h"

#include "Framework/CudaEventBenchmark.h"

//-----------------------
//--- cudaRuntimeTest ---
//-----------------------

namespace {

class cudaRuntimeImpl : public CudaEventBenchmark<size_t> {
  public:
    cudaRuntimeImpl() = default;

    virtual std::string Name() const override { return "bandwidth::cudaRuntime()"; }

    virtual void SingleRun() override {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
};

} // unnamed namespace


namespace bandwidth {

std::unique_ptr<IMicrobenchmark<size_t>> cudaRuntimeTest() {
  return std::make_unique<cudaRuntimeImpl>();
}

}
