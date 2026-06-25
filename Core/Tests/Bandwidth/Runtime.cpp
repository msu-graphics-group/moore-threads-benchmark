#include "Tests/Bandwidth.h"

#include "Framework/CudaEventBenchmark.h"


//----------------------
//--- cudaMemcpyTest ---
//----------------------

namespace {

class cudaMemcpyImpl : public CudaEventBenchmark<size_t> {
  public:
    cudaMemcpyImpl() = default;

    virtual void Init() override {
      auto block_size = std::get<0>(Args());
      host_ = std::make_unique<char[]>(block_size);
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
    std::unique_ptr<char[]> host_;
    char *device_src_{};
    char *device_dst_{};
};

// Host to Device
class cudaMemcpyHostToDeviceImpl : public cudaMemcpyImpl {
  public:
    cudaMemcpyHostToDeviceImpl() = default;

    virtual std::string Name() const override { return "bandwidth::cudaMemcpyHostToDevice()";}

    virtual void SingleRun() override {
      auto block_size = std::get<0>(Args());
      for(size_t j = 0; j < SubIterations(); j++) {
        HANDLE_ERROR(Api::cudaMemcpy(device_dst_, host_.get(), block_size,
                                     Api::cudaMemcpyHostToDevice));
      }
    }
};

// Device to Host
class cudaMemcpyDeviceToHostImpl : public cudaMemcpyImpl {
  public:
    cudaMemcpyDeviceToHostImpl() = default;

    virtual std::string Name() const override { return "bandwidth::cudaMemcpyDeviceToHost()";}

    virtual void SingleRun() override {
      auto block_size = std::get<0>(Args());
      for(size_t j = 0; j < SubIterations(); j++) {
        HANDLE_ERROR(Api::cudaMemcpy(host_.get(), device_src_, block_size,
                                     Api::cudaMemcpyDeviceToHost));
      }
    }
};

// Device to Device
class cudaMemcpyDeviceToDeviceImpl : public cudaMemcpyImpl {
  public:
    cudaMemcpyDeviceToDeviceImpl() = default;

    virtual std::string Name() const override { return "bandwidth::cudaMemcpyDeviceToDevice()";}

    virtual void SingleRun() override {
      auto block_size = std::get<0>(Args());
      for(size_t j = 0; j < SubIterations(); j++) {
        HANDLE_ERROR(Api::cudaMemcpy(device_dst_, device_src_, block_size,
                                     Api::cudaMemcpyDeviceToDevice));
      }
    }
};

} // unnamed namespace


namespace bandwidth {

std::unique_ptr<IMicrobenchmark<size_t>> cudaMemcpyHostToDeviceTest() {
  return std::make_unique<cudaMemcpyHostToDeviceImpl>();
}

std::unique_ptr<IMicrobenchmark<size_t>> cudaMemcpyDeviceToHostTest() {
  return std::make_unique<cudaMemcpyDeviceToHostImpl>();
}

std::unique_ptr<IMicrobenchmark<size_t>> cudaMemcpyDeviceToDeviceTest() {
  return std::make_unique<cudaMemcpyDeviceToDeviceImpl>();
}
} // namespace bandwidth

//----------------------------
//--- cudaMemcpyPinnedTest ---
//----------------------------

namespace {

struct CudaFreeHostDeleter {
  void operator()(char *ptr) const noexcept {
    if (ptr) {
      HANDLE_ERROR(Api::cudaFreeHost(ptr));
    }
  }
};

using PinnedHostPtr = std::unique_ptr<char, CudaFreeHostDeleter>;

class cudaMemcpyPinnedImpl : public CudaEventBenchmark<size_t> {
  public:
    cudaMemcpyPinnedImpl() = default;

    virtual void Init() override {
      auto block_size = std::get<0>(Args());
      char *host{};
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
    PinnedHostPtr host_;
    char *device_{};
};

// Host To Device
class cudaMemcpyPinnedHostToDeviceImpl: public cudaMemcpyPinnedImpl {
  public:
    cudaMemcpyPinnedHostToDeviceImpl() = default;

    virtual std::string Name() const override { return"bandwidth::cudaMemcpyPinnedHostToDevice()"; }

    virtual void SingleRun() override {
      auto block_size = std::get<0>(Args());
      for (size_t j = 0; j < SubIterations(); j++) {
        HANDLE_ERROR(Api::cudaMemcpy(device_, host_.get(), block_size,
                                     Api::cudaMemcpyHostToDevice));
      }
    }
};

// Device To Host
class cudaMemcpyPinnedDeviceToHostImpl: public cudaMemcpyPinnedImpl {
  public:
    cudaMemcpyPinnedDeviceToHostImpl() = default;

    virtual std::string Name() const override { return"bandwidth::cudaMemcpyPinnedDeviceToHost()"; }

    virtual void SingleRun() override {
      auto block_size = std::get<0>(Args());
      for (size_t j = 0; j < SubIterations(); j++) {
        HANDLE_ERROR(Api::cudaMemcpy(host_.get(), device_, block_size,
                                     Api::cudaMemcpyDeviceToHost));
      }
    }
};

} // unnamed namespace

namespace bandwidth {

std::unique_ptr<IMicrobenchmark<size_t>> cudaMemcpyPinnedHostToDeviceTest() {
  return std::make_unique<cudaMemcpyPinnedHostToDeviceImpl>();
}

std::unique_ptr<IMicrobenchmark<size_t>> cudaMemcpyPinnedDeviceToHostTest() {
  return std::make_unique<cudaMemcpyPinnedDeviceToHostImpl>();
}

} // namespace bandwidth

//----------------------------
//--- cudaMemcpyManagedTest --
//----------------------------

namespace {

class cudaMemcpyManagedImpl : public CudaEventBenchmark<size_t> {
  public:
    cudaMemcpyManagedImpl() = default;

    virtual void Init() override {
      auto block_size = std::get<0>(Args());
      HANDLE_ERROR(Api::cudaMallocManaged(&managed_, block_size));
      HANDLE_ERROR(Api::cudaMalloc(&device_, block_size));
    }

    virtual void CleanUp() override {
      HANDLE_ERROR(Api::cudaFree(managed_));
      HANDLE_ERROR(Api::cudaFree(device_));
      managed_ = nullptr;
      device_ = nullptr;
    }

  protected:
    char *managed_{};
    char *device_{};
};

// Managed To Device
class cudaMemcpyManagedToDeviceImpl : public cudaMemcpyManagedImpl {
  public:
    cudaMemcpyManagedToDeviceImpl() = default;

    virtual std::string Name() const override { return "bandwidth::cudaMemcpyManagedToDevice()";}

    virtual void SingleRun() override {
      auto block_size = std::get<0>(Args());
      for(size_t j = 0; j < SubIterations(); j++) {
        HANDLE_ERROR(Api::cudaMemcpy(device_, managed_, block_size,
                                     Api::cudaMemcpyHostToDevice));
      }
    }
};

// Device To Managed
class cudaMemcpyDeviceToManagedImpl : public cudaMemcpyManagedImpl {
  public:
    cudaMemcpyDeviceToManagedImpl() = default;

    virtual std::string Name() const override { return "bandwidth::cudaMemcpyDeviceToManaged()";}

    virtual void SingleRun() override {
      auto block_size = std::get<0>(Args());
      for(size_t j = 0; j < SubIterations(); j++) {
        HANDLE_ERROR(Api::cudaMemcpy(managed_, device_, block_size,
                                     Api::cudaMemcpyDeviceToHost));
      }
    }
};

} // unnamed namespace

namespace bandwidth {

std::unique_ptr<IMicrobenchmark<size_t>> cudaMemcpyManagedToDeviceTest() {
  return std::make_unique<cudaMemcpyManagedToDeviceImpl>();
}

std::unique_ptr<IMicrobenchmark<size_t>> cudaMemcpyDeviceToManagedTest() {
  return std::make_unique<cudaMemcpyDeviceToManagedImpl>();
}

} // namespace bandwidth
