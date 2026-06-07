#include "Tests/Overheads.h"

#include "Microbenchmarks/CudaEventBenchmark.h"

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

//-----------------------------
//--- cudaMallocManagedTest ---
//-----------------------------

namespace {

class cudaMallocManagedImpl: public CudaEventBenchmark<size_t> {
  public:
    cudaMallocManagedImpl() = default;

    virtual std::string Name() const override { return "overheads::cudaMallocManaged()";}

    virtual void Init() override {
      allocations.reserve(SubIterations());
    }

    virtual void SingleRun() override {
      auto block_size = std::get<0>(Args());
      for(size_t j = 0; j < SubIterations(); j++) {
        void *ptr{};
        HANDLE_ERROR(Api::cudaMallocManaged(&ptr, block_size));
        allocations.emplace_back(ptr);
      }
    }

    virtual void CleanUp() override {
      for(auto ptr : allocations) {
        HANDLE_ERROR(Api::cudaFree(ptr));
      }
      allocations.clear();
    }

  private:
    std::vector<void *> allocations;
};

} // unnamed namespace

namespace overheads {

std::unique_ptr<IMicrobenchmark<size_t>> cudaMallocManagedTest() {
  return std::make_unique<cudaMallocManagedImpl>();
}

} // namespace overheads

//--------------------------
//--- cudaMallocHostTest ---
//--------------------------

namespace {

class cudaMallocHostImpl: public CudaEventBenchmark<size_t> {
  public:
    cudaMallocHostImpl() = default;

    virtual std::string Name() const override { return "overheads::cudaMallocHost()";}

    virtual void Init() {
      allocations.reserve(SubIterations());
    }

    virtual void SingleRun() {
      auto block_size = std::get<0>(Args());
      for(size_t j = 0; j < SubIterations(); j++) {
        void *ptr{};
        HANDLE_ERROR(Api::cudaMallocHost(&ptr, block_size));
        allocations.emplace_back(ptr);
      }
    }

    virtual void CleanUp() override {
      for(auto ptr : allocations) {
        HANDLE_ERROR(Api::cudaFreeHost(ptr));
      }
      allocations.clear();
    }

  private:
    std::vector<void *> allocations;
};

} // unnamed namespace

namespace overheads {

std::unique_ptr<IMicrobenchmark<size_t>> cudaMallocHostTest() {
  return std::make_unique<cudaMallocHostImpl>();
}

} //namespace overheads

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

//------------------------
//--- cudaFreeHostTest ---
//------------------------

namespace {

class cudaFreeHostImpl: public CudaEventBenchmark<size_t> {
  public:
    cudaFreeHostImpl() = default;

    virtual std::string Name() const override { return "overheads::cudaFreeHost()";}

    virtual void Init() override {
      auto block_size = std::get<0>(Args());
      for(size_t i = 0; i < Iterations(); i ++) {
        void *ptr{};
        HANDLE_ERROR(Api::cudaMallocHost(&ptr, block_size));
        allocations.emplace_back(ptr);
      }
    }

    virtual void SingleRun() override {
      for (auto ptr : allocations){
        HANDLE_ERROR(Api::cudaFreeHost(ptr));
      }
    }

    virtual void CleanUp() override {
      allocations.clear();
    }

    private:
      std::vector<void *> allocations;
};

} // unnamed namespace

namespace overheads {

std::unique_ptr<IMicrobenchmark<size_t>> cudaFreeHostTest() {
  return std::make_unique<cudaFreeHostImpl>();
}

} // namespace overheads

//----------------------
//--- cudaMemsetTest ---
//----------------------

namespace {

class cudaMemsetImpl: public CudaEventBenchmark<size_t> {
  public:
    cudaMemsetImpl() = default;

    virtual std::string Name() const override { return "overheads::cudaMemset()";}

    virtual void Init() override {
      auto block_size = std::get<0>(Args());
      HANDLE_ERROR(Api::cudaMalloc(&device_, block_size));
    }

    virtual void SingleRun() override {
      auto block_size = std::get<0>(Args());
      for(size_t j = 0; j < SubIterations(); j++) {
        HANDLE_ERROR(Api::cudaMemset(device_, 0, block_size));
      }
    }

    virtual void CleanUp() override {
      HANDLE_ERROR(Api::cudaFree(device_));
      device_ = nullptr;
    }

  private:
    void  *device_{};
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

// Public base
class cudaMemcpyImpl: public CudaEventBenchmark<size_t>{
  public:
    cudaMemcpyImpl() = default;

    virtual void Init() override{
      auto block_size = std::get<0>(Args());
      host_ = new char[block_size];
      HANDLE_ERROR(Api::cudaMalloc(&device_src_, block_size));
      HANDLE_ERROR(Api::cudaMalloc(&device_dst_, block_size));
    }

    virtual void CleanUp() override {
      HANDLE_ERROR(Api::cudaFree(device_src_));
      HANDLE_ERROR(Api::cudaFree(device_dst_));
      delete[] host_;

      device_src_ = nullptr;
      device_dst_ = nullptr;
      host_ = nullptr;
    }

  protected:
    char* host_ {};
    char* device_src_ {};
    char* device_dst_ {};
};

// Host to Device
class cudaMemcpyHostToDeviceImpl: public cudaMemcpyImpl {
  public:
    cudaMemcpyHostToDeviceImpl() = default;

    virtual std::string Name() const override { return "overheads::cudaMemcpyHostToDevice()";}

    virtual void SingleRun() override {
      auto block_size = std::get<0>(Args());
      for(size_t j = 0; j < SubIterations(); j++){
        HANDLE_ERROR(Api::cudaMemcpy(device_dst_, host_, block_size,
                                     Api::cudaMemcpyHostToDevice));
      }
    }
};

// Device to Host
class cudaMemcpyDeviceToHostImpl: public cudaMemcpyImpl {
  public:
    cudaMemcpyDeviceToHostImpl() = default;

    virtual std::string Name() const override { return "overheads::cudaMemcpyDeviceToHost()";}

    virtual void SingleRun() override {
      auto block_size = std::get<0>(Args());
      for(size_t j = 0; j < SubIterations(); j++){
        HANDLE_ERROR(Api::cudaMemcpy(host_, device_src_, block_size,
                                     Api::cudaMemcpyDeviceToHost));
      }
    }
};

// Device to Device
class cudaMemcpyDeviceToDeviceImpl: public cudaMemcpyImpl {
  public:
    cudaMemcpyDeviceToDeviceImpl() = default;

    virtual std::string Name() const override { return "overheads::cudaMemcpyDeviceToDevice()";}

    virtual void SingleRun() override {
      auto block_size = std::get<0>(Args());
      for(size_t j = 0; j < SubIterations(); j++){
        HANDLE_ERROR(Api::cudaMemcpy(device_dst_, device_src_, block_size,
                                    Api::cudaMemcpyDeviceToDevice));
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

//---------------------------
//--- cudaMemcpyAsyncTest ---
//---------------------------

namespace {

class cudaMemcpyAsyncImpl: public CudaEventBenchmark<size_t> {
  public:
    cudaMemcpyAsyncImpl() = default;

    virtual void Init() override {
      auto block_size = std::get<0>(Args());
      HANDLE_ERROR(Api::cudaMallocHost(&host_, block_size));
      HANDLE_ERROR(Api::cudaMalloc(&device_, block_size));
    }

    virtual void CleanUp() override {
      HANDLE_ERROR(Api::cudaFree(device_));
      HANDLE_ERROR(Api::cudaFreeHost(host_));
      device_ = nullptr;
      host_ = nullptr;
    }

  protected:
    char *host_{};
    char *device_{};
};

// Host to Device
class cudaMemcpyAsyncHostToDeviceImpl: public cudaMemcpyAsyncImpl {
  public:
    cudaMemcpyAsyncHostToDeviceImpl() = default;

    virtual std::string Name() const override { return"overheads::cudaMemcpyAsyncHostToDevice()";}

    virtual void SingleRun() override {
      auto block_size = std::get<0>(Args());
      for(size_t j = 0; j < SubIterations(); j++) {
        HANDLE_ERROR(Api::cudaMemcpyAsync(device_, host_, block_size,
                                          Api::cudaMemcpyHostToDevice));
      }
      HANDLE_ERROR(Api::cudaDeviceSynchronize());
    }
};

// Device to Host
class cudaMemcpyAsyncDeviceToHostImpl: public cudaMemcpyAsyncImpl {
  public:
    cudaMemcpyAsyncDeviceToHostImpl() = default;

    virtual std::string Name() const override { return"overheads::cudaMemcpyAsyncDeviceToHost()";}

    virtual void SingleRun() override {
      auto block_size = std::get<0>(Args());
      for(size_t j = 0; j < SubIterations(); j++) {
        HANDLE_ERROR(Api::cudaMemcpyAsync(host_, device_, block_size,
                                          Api::cudaMemcpyDeviceToHost));
      }
      HANDLE_ERROR(Api::cudaDeviceSynchronize());
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

} // namespace overheads

//----------------------------
//--- cudaMemcpyPinnedTest ---
//----------------------------

namespace {

class cudaMemcpyPinnedImpl: public CudaEventBenchmark<size_t> {
  public:
    cudaMemcpyPinnedImpl() = default;

    virtual void Init() override {
      auto block_size = std::get<0>(Args());
      HANDLE_ERROR(Api::cudaMallocHost(&host_, block_size));
      HANDLE_ERROR(Api::cudaMalloc(&device_, block_size));
    }

    virtual void CleanUp() override {
      HANDLE_ERROR(Api::cudaFreeHost(host_));
      HANDLE_ERROR(Api::cudaFree(device_));
      host_ = nullptr;
      device_ = nullptr;
    }

  protected:
    char* host_{};
    char* device_{};
};

// Host To Device
class cudaMemcpyPinnedHostToDeviceImpl: public cudaMemcpyPinnedImpl {
  public:
    cudaMemcpyPinnedHostToDeviceImpl() = default;

    virtual std::string Name() const override { return"overheads::cudaMemcpyPinnedHostToDevice()";}

    virtual void SingleRun() override {
      auto block_size = std::get<0>(Args());
      for(size_t j = 0; j < SubIterations(); j++) {
        HANDLE_ERROR(Api::cudaMemcpy(device_, host_, block_size,
                                     Api::cudaMemcpyHostToDevice));
      }
    }
};

//Device To Host
class cudaMemcpyPinnedDeviceToHostImpl: public cudaMemcpyPinnedImpl {
  public:
    cudaMemcpyPinnedDeviceToHostImpl() = default;

    virtual std::string Name() const override { return"overheads::cudaMemcpyPinnedDeviceToHost()";}

    virtual void SingleRun() override {
      auto block_size = std::get<0>(Args());
      for(size_t j = 0; j < SubIterations(); j++) {
        HANDLE_ERROR(Api::cudaMemcpy(host_, device_, block_size,
                                     Api::cudaMemcpyDeviceToHost));
      }
    }
};

} // unnamed namespace

namespace overheads {

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

namespace{

class cudaEventCreateImpl: public CudaEventBenchmark<>{
  public:
    cudaEventCreateImpl() = default;

    virtual std::string Name() const override{ return "overheads::cudaEventCreate()";}

    virtual void Init() override {
      events.reserve(SubIterations());
    }

    virtual void SingleRun() override{
      for(size_t j = 0; j < SubIterations(); j++){
        Api::cudaEvent_t event{};
        HANDLE_ERROR(Api::cudaEventCreate(&event));
        events.emplace_back(event);
      }
    }

    virtual void CleanUp() override{
      for(auto event : events){
        HANDLE_ERROR(Api::cudaEventDestroy(event));
      }
      events.clear();
    }

    private:
      std::vector<Api::cudaEvent_t> events;
};

} //unnamed namespace

namespace overheads{

std::unique_ptr<IMicrobenchmark<>> cudaEventCreateTest(){
  return std::make_unique<cudaEventCreateImpl>();
}

} // namespace overheads

//----------------------------
//--- cudaEventDestroyTest ---
//----------------------------

namespace{

class cudaEventDestroyImpl: public CudaEventBenchmark<>{
  public:
    cudaEventDestroyImpl() = default;

    virtual std::string Name() const override{ return "overheads::cudaEventDestroy()";}

    virtual void Init() override{
      events.reserve(SubIterations());
      for(size_t j = 0; j < SubIterations(); j++){
        Api::cudaEvent_t event{};
        HANDLE_ERROR(Api::cudaEventCreate(&event));
        events.emplace_back(event);
      }
    }

    virtual void SingleRun() override{
      for(auto event : events){
        HANDLE_ERROR(Api::cudaEventDestroy(event));
      }
    }

    virtual void CleanUp() override{
      events.clear();
    }
  private:
    std::vector<Api::cudaEvent_t> events;
};

} // unnamed namespace

namespace overheads{

std::unique_ptr<IMicrobenchmark<>> cudaEventDestroyTest(){
  return std::make_unique<cudaEventDestroyImpl>();
}

} // namespace overheads

//---------------------------
//--- cudaEventRecordTest ---
//---------------------------

namespace {

class cudaEventRecordImpl: public CudaEventBenchmark<>{
  public:
    cudaEventRecordImpl() = default;

    virtual std::string Name() const override { return "overheads::cudaEventRecord()";}

    virtual void Init() override{
      HANDLE_ERROR(Api::cudaEventCreate(&event));
    }

    virtual void SingleRun() override{
      for(size_t j = 0; j < SubIterations(); j++){
        HANDLE_ERROR(Api::cudaEventRecord(event));
      }
    }

    virtual void CleanUp() override{
      HANDLE_ERROR(Api::cudaEventDestroy(event));
    }

  private:
    Api::cudaEvent_t event{};
};

} //unnamed namespace

namespace overheads {

std::unique_ptr<IMicrobenchmark<>> cudaEventRecordTest(){
  return std::make_unique<cudaEventRecordImpl>();
}

} // namespace overheads


//---------------------------------
//--- cudaDeviceSynchronizeTest ---
//---------------------------------

namespace {
class cudaDeviceSynchronizeImpl: public CudaEventBenchmark<>{
  public:
    cudaDeviceSynchronizeImpl() = default;

    virtual std::string Name() const override{ return "overheads::cudaDeviceSynchronize()";}

    virtual void SingleRun() override {
      for(size_t j = 0; j < SubIterations(); j++){
        HANDLE_ERROR(Api::cudaDeviceSynchronize());
      }
    }
};

} // unnamed namespace

namespace overheads {

std::unique_ptr<IMicrobenchmark<>> cudaDeviceSynchronizeTest(){
  return std::make_unique<cudaDeviceSynchronizeImpl>();
}

} // namespace overheads