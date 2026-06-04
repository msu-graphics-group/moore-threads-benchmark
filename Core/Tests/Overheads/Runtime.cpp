#include "Tests/Overheads.h"

#include "Framework/CudaEventBenchmark.h"

//----------------------
//--- cudaMallocTest ---
//----------------------

namespace {

class cudaMallocImpl : public CudaEventBenchmark<size_t> {
  public:
    cudaMallocImpl() = default;

    virtual std::string Name() const override { return "overheads::cudaMalloc()"; }

    virtual void Init() override {
      allocations.reserve(SubIterations());
    }

    virtual void SingleRun() override {
      auto block_size = std::get<0>(Args());
      for (size_t j = 0; j < SubIterations(); j++) {
        void *ptr{};
        HANDLE_ERROR(Api::cudaMalloc(&ptr, block_size));
        allocations.emplace_back(ptr);
      }
    }

    virtual void CleanUp() override {
      for (auto ptr : allocations) {
        HANDLE_ERROR(Api::cudaFree(ptr));
      }
      allocations.clear();
    }

  private:
    std::vector<void *> allocations;
};

} // unnamed namespace

namespace overheads {

std::unique_ptr<IMicrobenchmark<size_t>> cudaMallocTest() {
  return std::make_unique<cudaMallocImpl>();
}

} // namespace overheads


//--------------------
//--- cudaFreeTest ---
//--------------------

namespace {

class cudaFreeImpl : public CudaEventBenchmark<size_t> {
  public:
    cudaFreeImpl() = default;

    virtual std::string Name() const override { return "overheads::cudaFree()"; }

    virtual void Init() override {
      auto block_size = std::get<0>(Args());
      for (size_t i = 0; i < Iterations(); i++) {
        void *ptr{};
        HANDLE_ERROR(Api::cudaMalloc(&ptr, block_size));
        allocations.emplace_back(ptr);
      }
    }

    virtual void SingleRun() override {
      for (auto ptr : allocations) {
        HANDLE_ERROR(Api::cudaFree(ptr));
      }
    }

    virtual void CleanUp() override { allocations.clear(); }

  private:
    std::vector<void *> allocations;
};

} // unnamed namespace


namespace overheads {

std::unique_ptr<IMicrobenchmark<size_t>> cudaFreeTest() {
  return std::make_unique<cudaFreeImpl>();
}

} // namespace overheads
