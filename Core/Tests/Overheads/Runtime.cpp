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