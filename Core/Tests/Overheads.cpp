#include "Tests/Overheads.h"

#include "Framework/CudaEventBenchmark.h"


using cudaMallocType = Api::cudaError_t (*)(void **ptr, size_t size);
using cudaFreeType   = Api::cudaError_t (*)(void *ptr);

namespace {

size_t RandomOffset(const std::tuple<size_t> &args) {
  auto block_size = std::get<0>(args);
  assert(block_size > 1);
  return rand() % (block_size - 1);
}

} // unnamed namespace

//-----------------------
//--- MallocBenchmark ---
//-----------------------

namespace {

template <typename AllocFn, typename FreeFn>
class MallocBenchmark : public CudaEventBenchmark<size_t> {
  public:
    MallocBenchmark(const std::string &name, AllocFn alloc_fn, FreeFn free_fn)
      : name_(name), alloc_(alloc_fn), free_(free_fn) {
    }

    virtual std::string Name() const override { return name_; }

    virtual void Init() override {
      allocations_.reserve(SubIterations());
    }

    virtual void SingleRun() override {
      auto block_size = std::get<0>(Args());
      for (size_t j = 0; j < SubIterations(); j++) {
        void *ptr{};
        HANDLE_ERROR(alloc_(&ptr, block_size));
        allocations_.emplace_back(ptr);
      }
    }

    virtual void CleanUp() override {
      for (auto ptr : allocations_) {
        HANDLE_ERROR(free_(ptr));
      }
      allocations_.clear();
    }

  private:
    std::string name_;
    AllocFn alloc_;
    FreeFn free_;
    std::vector<void *> allocations_;
};

} // unnamed namespace

namespace overheads {

std::unique_ptr<IMicrobenchmark<size_t>> cudaMallocTest() {
  return std::unique_ptr<IMicrobenchmark<size_t>>(
    new MallocBenchmark<cudaMallocType, cudaFreeType>(
      "overheads::cudaMalloc()",
      Api::cudaMalloc,
      Api::cudaFree)
  );
}

std::unique_ptr<IMicrobenchmark<size_t>> cudaMallocManagedTest() {
  return std::unique_ptr<IMicrobenchmark<size_t>>(
    new MallocBenchmark<cudaMallocType, cudaFreeType>(
      "overheads::cudaMallocManaged()",
      Api::cudaMallocManaged,
      Api::cudaFree)
  );
}

std::unique_ptr<IMicrobenchmark<size_t>> cudaHostAllocTest() {
  return std::unique_ptr<IMicrobenchmark<size_t>>(
    new MallocBenchmark(
      "overheads::cudaHostAlloc()",
      [](void **ptr, size_t size) { return Api::cudaHostAlloc(ptr, size, Api::cudaHostAllocDefault); },
      [](void *ptr) { return Api::cudaFreeHost(ptr); })
  );
}

} // namespace overheads


//---------------------
//--- FreeBenchmark ---
//---------------------

namespace {

template <typename AllocFn, typename FreeFn>
class FreeBenchmark : public CudaEventBenchmark<size_t> {
  public:
    FreeBenchmark(const std::string &name, AllocFn alloc_fn, FreeFn free_fn)
      : name_(name), alloc_(alloc_fn), free_(free_fn) {
    }

    virtual std::string Name() const override { return name_; }

    virtual void Init() override {
      auto block_size = std::get<0>(Args());
      for (size_t i = 0; i < SubIterations(); i++) {
        void *ptr{};
        HANDLE_ERROR(alloc_(&ptr, block_size));
        allocations_.emplace_back(ptr);
      }
    }

    virtual void SingleRun() override {
      for (auto ptr : allocations_) {
        HANDLE_ERROR(free_(ptr));
      }
    }

    virtual void CleanUp() override { allocations_.clear(); }

  private:
    std::string name_;
    AllocFn alloc_;
    FreeFn free_;
    std::vector<void *> allocations_;
};

} // unnamed namespace


namespace overheads {

std::unique_ptr<IMicrobenchmark<size_t>> cudaFreeTest() {
  return std::unique_ptr<IMicrobenchmark<size_t>>(
    new FreeBenchmark<cudaMallocType, cudaFreeType>(
      "overheads::cudaFree()",
      Api::cudaMalloc,
      Api::cudaFree)
  );
}

std::unique_ptr<IMicrobenchmark<size_t>> cudaFreeHostTest() {
  return std::unique_ptr<IMicrobenchmark<size_t>>(
    new FreeBenchmark(
      "overheads::cudaFreeHost()",
      [](void **ptr, size_t size) { return Api::cudaHostAlloc(ptr, size, Api::cudaHostAllocDefault); },
      [](void *ptr) { return Api::cudaFreeHost(ptr); })
  );
}

} // namespace overheads


//----------------------
//--- cudaMemsetTest ---
//----------------------

namespace {

class cudaMemsetImpl: public CudaEventBenchmark<size_t> {
  public:
    cudaMemsetImpl() = default;

    virtual std::string Name() const override { return "overheads::cudaMemset()"; }

    virtual void Init() override {
      auto block_size = std::get<0>(Args());
      HANDLE_ERROR(Api::cudaMalloc(&device_, block_size));
    }

    virtual void SingleRun() override {
      for (size_t j = 0; j < SubIterations(); j++) {
        HANDLE_ERROR(Api::cudaMemset(device_ + RandomOffset(Args()), 0, 1));
      }
    }

    virtual void CleanUp() override {
      HANDLE_ERROR(Api::cudaFree(device_));
      device_ = nullptr;
    }

  private:
    std::byte *device_{};
};

} // unnamed namespace

namespace overheads {

std::unique_ptr<IMicrobenchmark<size_t>> cudaMemsetTest() {
  return std::make_unique<cudaMemsetImpl>();
}

} // namespace overheads


//----------------------
//--- cudaMemcpyTest ---
//----------------------

namespace {

// Base class for cudaMemcpy benchmarks
class cudaMemcpyImpl: public CudaEventBenchmark<size_t> {
  public:
    cudaMemcpyImpl() = default;

    virtual void Init() override{
      auto block_size = std::get<0>(Args());
      host_ = std::make_unique<std::byte[]>(block_size);
      HANDLE_ERROR(Api::cudaMalloc(&device_src_, block_size));
      HANDLE_ERROR(Api::cudaMalloc(&device_dst_, block_size));
    }

    virtual void CleanUp() override {
      HANDLE_ERROR(Api::cudaFree(device_src_));
      HANDLE_ERROR(Api::cudaFree(device_dst_));
      host_.reset();

      device_src_ = nullptr;
      device_dst_ = nullptr;
    }

  protected:  
    std::unique_ptr<std::byte[]> host_;
    std::byte *device_src_{};
    std::byte *device_dst_{};
};

// Host to Device
class cudaMemcpyHostToDeviceImpl: public cudaMemcpyImpl {
  public:
    cudaMemcpyHostToDeviceImpl() = default;

    virtual std::string Name() const override { return "overheads::cudaMemcpy()::host->device"; }

    virtual void SingleRun() override {
      for(size_t j = 0; j < SubIterations(); j++){
        HANDLE_ERROR(Api::cudaMemcpy(device_dst_ + RandomOffset(Args()),
                                     host_.get() + RandomOffset(Args()),
                                     1, Api::cudaMemcpyHostToDevice));
      }
    }
};

// Device to Host
class cudaMemcpyDeviceToHostImpl: public cudaMemcpyImpl {
  public:
    cudaMemcpyDeviceToHostImpl() = default;

    virtual std::string Name() const override { return "overheads::cudaMemcpy()::device->host"; }

    virtual void SingleRun() override {
      for(size_t j = 0; j < SubIterations(); j++){
        HANDLE_ERROR(Api::cudaMemcpy(host_.get() + RandomOffset(Args()),
                                     device_src_ + RandomOffset(Args()),
                                     1, Api::cudaMemcpyDeviceToHost));
      }
    }
};

// Device to Device
class cudaMemcpyDeviceToDeviceImpl: public cudaMemcpyImpl {
  public:
    cudaMemcpyDeviceToDeviceImpl() = default;

    virtual std::string Name() const override { return "overheads::cudaMemcpy()::device->device"; }

    virtual void SingleRun() override {
      for(size_t j = 0; j < SubIterations(); j++){
        HANDLE_ERROR(Api::cudaMemcpy(device_dst_ + RandomOffset(Args()),
                                     device_src_ + RandomOffset(Args()),
                                     1, Api::cudaMemcpyDeviceToDevice));
      }
    }
};

} // unnamed namespace

namespace overheads {

std::unique_ptr<IMicrobenchmark<size_t>> cudaMemcpyHostToDeviceTest() {
  return std::make_unique<cudaMemcpyHostToDeviceImpl>();
}

std::unique_ptr<IMicrobenchmark<size_t>> cudaMemcpyDeviceToHostTest() {
  return std::make_unique<cudaMemcpyDeviceToHostImpl>();
}

std::unique_ptr<IMicrobenchmark<size_t>> cudaMemcpyDeviceToDeviceTest() {
  return std::make_unique<cudaMemcpyDeviceToDeviceImpl>();
}

} // namespace overheads


//-----------------------------
//--- PinnedMemoryBenchmark ---
//-----------------------------

namespace {

// Coded by DeepSeek-v4
struct CudaFreeHostDeleter {
  void operator()(std::byte *ptr) const {
    if (ptr) {
      HANDLE_ERROR(Api::cudaFreeHost(ptr));
    }
  }
};

class PinnedMemoryBenchmark: public CudaEventBenchmark<size_t> {
  public:
    PinnedMemoryBenchmark() = default;

    virtual void Init() override {
      auto block_size = std::get<0>(Args());
      std::byte *host{};
      HANDLE_ERROR(Api::cudaHostAlloc(&host, block_size, Api::cudaHostAllocDefault));
      host_.reset(host);
      HANDLE_ERROR(Api::cudaMalloc(&device_, block_size));
    }

    virtual void CleanUp() override {
      HANDLE_ERROR(Api::cudaFree(device_));
      host_.reset();
      device_ = nullptr;
    }

  protected:
    std::unique_ptr<std::byte, CudaFreeHostDeleter> host_{};
    std::byte *device_{};
};

// Host to Device, async
class cudaMemcpyAsyncHostToDeviceImpl: public PinnedMemoryBenchmark {
  public:
    cudaMemcpyAsyncHostToDeviceImpl() = default;

    virtual std::string Name() const override { return"overheads::cudaMemcpyAsync()::host->device"; }

    virtual void SingleRun() override {
      for(size_t j = 0; j < SubIterations(); j++) {
        HANDLE_ERROR(Api::cudaMemcpyAsync(device_ + RandomOffset(Args()),
                                          host_.get() + RandomOffset(Args()),
                                          1, Api::cudaMemcpyHostToDevice));
      }
    }
};

// Device to Host, async
class cudaMemcpyAsyncDeviceToHostImpl: public PinnedMemoryBenchmark {
  public:
    cudaMemcpyAsyncDeviceToHostImpl() = default;

    virtual std::string Name() const override { return"overheads::cudaMemcpyAsync()::device->host";}

    virtual void SingleRun() override {
      for(size_t j = 0; j < SubIterations(); j++) {
        HANDLE_ERROR(Api::cudaMemcpyAsync(host_.get() + RandomOffset(Args()),
                                          device_ + RandomOffset(Args()),
                                          1, Api::cudaMemcpyDeviceToHost));
      }
    }
};

// Host To Device, sync
class cudaMemcpyPinnedHostToDeviceImpl: public PinnedMemoryBenchmark {
  public:
    cudaMemcpyPinnedHostToDeviceImpl() = default;

    virtual std::string Name() const override { return"overheads::cudaMemcpy()::pinned::host->device"; }

    virtual void SingleRun() override {
      for (size_t j = 0; j < SubIterations(); j++) {
        HANDLE_ERROR(Api::cudaMemcpy(device_ + RandomOffset(Args()),
                                     host_.get() + RandomOffset(Args()), 1,
                                     Api::cudaMemcpyHostToDevice));
      }
    }
};

// Device To Host, sync
class cudaMemcpyPinnedDeviceToHostImpl: public PinnedMemoryBenchmark {
  public:
    cudaMemcpyPinnedDeviceToHostImpl() = default;

    virtual std::string Name() const override { return"overheads::cudaMemcpy()::pinned::device->host"; }

    virtual void SingleRun() override {
      for (size_t j = 0; j < SubIterations(); j++) {
        HANDLE_ERROR(Api::cudaMemcpy(host_.get() + RandomOffset(Args()),
                                     device_ + RandomOffset(Args()), 1,
                                     Api::cudaMemcpyDeviceToHost));
      }
    }
};

} // unnamed namespace

namespace overheads {

std::unique_ptr<IMicrobenchmark<size_t>> cudaMemcpyAsyncHostToDeviceTest() {
  return std::make_unique<cudaMemcpyAsyncHostToDeviceImpl>();
}

std::unique_ptr<IMicrobenchmark<size_t>> cudaMemcpyAsyncDeviceToHostTest() {
  return std::make_unique<cudaMemcpyAsyncDeviceToHostImpl>();
}

std::unique_ptr<IMicrobenchmark<size_t>> cudaMemcpyPinnedHostToDeviceTest() {
  return std::make_unique<cudaMemcpyPinnedHostToDeviceImpl>();
}

std::unique_ptr<IMicrobenchmark<size_t>> cudaMemcpyPinnedDeviceToHostTest() {
  return std::make_unique<cudaMemcpyPinnedDeviceToHostImpl>();
}

} // namespace overheads


//---------------------------
//--- cudaEventCreateTest ---
//---------------------------

namespace {

class cudaEventCreateImpl: public CudaEventBenchmark<>{
  public:
    cudaEventCreateImpl() = default;

    virtual std::string Name() const override{ return "overheads::cudaEventCreate()"; }

    virtual void Init() override {
      events.reserve(SubIterations());
    }

    virtual void SingleRun() override {
      for (size_t j = 0; j < SubIterations(); j++) {
        Api::cudaEvent_t event{};
        HANDLE_ERROR(Api::cudaEventCreate(&event));
        events.emplace_back(event);
      }
    }

    virtual void CleanUp() override{
      for(auto event : events) {
        HANDLE_ERROR(Api::cudaEventDestroy(event));
      }
      events.clear();
    }

    private:
      std::vector<Api::cudaEvent_t> events;
};

} //unnamed namespace

namespace overheads {

std::unique_ptr<IMicrobenchmark<>> cudaEventCreateTest() {
  return std::make_unique<cudaEventCreateImpl>();
}

} // namespace overheads

//----------------------------
//--- cudaEventDestroyTest ---
//----------------------------

namespace {

class cudaEventDestroyImpl: public CudaEventBenchmark<> {
  public:
    cudaEventDestroyImpl() = default;

    virtual std::string Name() const override{ return "overheads::cudaEventDestroy()"; }

    virtual void Init() override {
      events.resize(SubIterations());
      for (size_t j = 0; j < events.size(); j++) {
        HANDLE_ERROR(Api::cudaEventCreate(events.data() + j));
      }
    }

    virtual void SingleRun() override {
      for (const auto &event : events) {
        HANDLE_ERROR(Api::cudaEventDestroy(event));
      }
    }

    virtual void CleanUp() override {
      events.clear();
    }

  private:
    std::vector<Api::cudaEvent_t> events;
};

} // unnamed namespace

namespace overheads {

std::unique_ptr<IMicrobenchmark<>> cudaEventDestroyTest() {
  return std::make_unique<cudaEventDestroyImpl>();
}

} // namespace overheads

//---------------------------
//--- cudaEventRecordTest ---
//---------------------------

namespace {

class cudaEventRecordImpl: public CudaEventBenchmark<> {
  public:
    cudaEventRecordImpl() = default;

    virtual std::string Name() const override { return "overheads::cudaEventRecord()"; }

    virtual void Init() override {
      for (size_t j = 0; j < SubIterations(); j++) {
        Api::cudaEvent_t event{};
        HANDLE_ERROR(Api::cudaEventCreate(&event));
        events.emplace_back(event);
      }
    }

    virtual void SingleRun() override {
      for (auto event : events) {
        HANDLE_ERROR(Api::cudaEventRecord(event));
      }
    }

    virtual void CleanUp() override {
      for (auto event : events) {
        HANDLE_ERROR(Api::cudaEventDestroy(event));
      }
      events.clear();
    }

  private:
    std::vector<Api::cudaEvent_t> events;
};

} //unnamed namespace

namespace overheads {

std::unique_ptr<IMicrobenchmark<>> cudaEventRecordTest() {
  return std::make_unique<cudaEventRecordImpl>();
}

} // namespace overheads


//---------------------------------
//--- cudaDeviceSynchronizeTest ---
//---------------------------------

namespace {

class cudaDeviceSynchronizeImpl : public CudaEventBenchmark<> {
  public:
    cudaDeviceSynchronizeImpl() = default;

    virtual std::string Name() const override{ return "overheads::cudaDeviceSynchronize()"; }

    virtual void SingleRun() override {
      for (size_t j = 0; j < SubIterations(); j++) {
        HANDLE_ERROR(Api::cudaDeviceSynchronize());
      }
    }
};

} // unnamed namespace

namespace overheads {

std::unique_ptr<IMicrobenchmark<>> cudaDeviceSynchronizeTest() {
  return std::make_unique<cudaDeviceSynchronizeImpl>();
}

} // namespace overheads


//---------------------------
//--- cudaDeviceResetTest ---
//---------------------------

namespace {

class cudaDeviceResetImpl : public IMicrobenchmark<> {
  public:
    cudaDeviceResetImpl() = default;

    virtual std::string Name() const override { return "overheads::cudaDeviceReset()"; }

    virtual void Configure(size_t iterations, size_t sub_iterations) override {
      assert(iterations >= 1);
      if (sub_iterations != 1) {
        throw std::invalid_argument("The number of sub-iterations must be equal to one");
      }
      iterations_ = iterations;
    }

    virtual std::vector<double> Run() override {
      using clock = std::chrono::steady_clock;

      std::vector<double> times;
      times.reserve(iterations_);
      for (size_t i = 0; i < iterations_; ++i) {
        auto start = clock::now();
        HANDLE_ERROR(Api::cudaDeviceReset());
        times.emplace_back(std::chrono::duration<double>(clock::now() - start).count());
      }
      return times;
    }

  private:
    size_t iterations_{};
};

} // unnamed namespace

namespace overheads {

std::unique_ptr<IMicrobenchmark<>> cudaDeviceResetTest() {
  return std::make_unique<cudaDeviceResetImpl>();
}

} // namespace overheads
