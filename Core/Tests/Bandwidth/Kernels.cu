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

//-------------------------------
//--- On-chip memory helpers  ---
//-------------------------------

// The idea is borrowed from the SHOC benchmark (src/cuda/level0/DeviceMemory.cu 516-579):
namespace {

constexpr int     kBlockThreads   = 256;
constexpr int     kBlocks         = 64;
constexpr size_t  kTotalThreads   = size_t(kBlocks) * kBlockThreads;
constexpr int     kUnroll         = 16;
constexpr size_t  kBytesPerRepeat = kTotalThreads * kUnroll * sizeof(float);

class KernelBandwidthImpl : public CudaEventBenchmark<size_t> {
  public:
    KernelBandwidthImpl() = default;

    virtual void Init() override {
      HANDLE_ERROR(Api::cudaMalloc(&output_, kTotalThreads * sizeof(float)));
    }

    virtual void CleanUp() override {
      HANDLE_ERROR(Api::cudaFree(output_));
      output_ = nullptr;
    }

  protected:
    size_t Repeats() const {
      auto block_size = std::get<0>(Args());
      assert(block_size >= kBytesPerRepeat && block_size % kBytesPerRepeat == 0);
      return block_size / kBytesPerRepeat;
    }
    float *output_ {};
};

} // unnamed namespace

//---------------------------
//--- sharedMemoryTest    ---
//---------------------------

namespace {

constexpr int kSharedFloats = 2048;

// Reads 'kUnroll' floats from shared memory per loop step and accumulates them
__global__ void SharedMemoryReadKernel(float *output, size_t repeats) {
  __shared__ float buffer[kSharedFloats];

  for (int i = threadIdx.x; i < kSharedFloats; i += blockDim.x) {
    buffer[i] = static_cast<float>(i);
  }
  __syncthreads();

  float sum = 0.0f;
  int index = threadIdx.x & (kSharedFloats - 1);
  for (size_t j = 0; j < repeats; j++) {
    #pragma unroll
    for (size_t k = 0; k < kUnroll; k++) {
      sum += buffer[(index + k) & (kSharedFloats - 1)];
    }
    index = (index + kUnroll) & (kSharedFloats - 1);
  }

  size_t gid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  output[gid] = sum;
}

// Writes 'kUnroll' floats to shared memory per loop step
__global__ void SharedMemoryWriteKernel(float *output, size_t repeats) {
  __shared__ float buffer[kSharedFloats];

  int index = threadIdx.x & (kSharedFloats - 1);
  for (size_t j = 0; j < repeats; j++) {
    float value = static_cast<float>(j);
    #pragma unroll
    for (size_t k = 0; k < kUnroll; k++) {
      buffer[(index + k) & (kSharedFloats - 1)] = value;
    }
    index = (index + kUnroll) & (kSharedFloats - 1);
  }
  __syncthreads();

  size_t gid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  output[gid] = buffer[threadIdx.x & (kSharedFloats - 1)];
}

// Read
class SharedMemoryReadImpl : public KernelBandwidthImpl {
  public:
    SharedMemoryReadImpl() = default;

    virtual std::string Name() const override { return "bandwidth::sharedMemoryRead()";}

    virtual void SingleRun() override {
      auto repeats = Repeats();
      for (size_t j = 0; j < SubIterations(); j++) {
        SharedMemoryReadKernel<<<kBlocks, kBlockThreads>>>(output_, repeats);
      }
      HANDLE_ERROR(Api::cudaGetLastError());
    }
};

// Write
class SharedMemoryWriteImpl : public KernelBandwidthImpl {
  public:
    SharedMemoryWriteImpl() = default;

    virtual std::string Name() const override { return "bandwidth::sharedMemoryWrite()";}

    virtual void SingleRun() override {
      auto repeats = Repeats();
      for (size_t j = 0; j < SubIterations(); j++) {
        SharedMemoryWriteKernel<<<kBlocks, kBlockThreads>>>(output_, repeats);
      }
      HANDLE_ERROR(Api::cudaGetLastError());
    }
};

} // unnamed namespace

namespace bandwidth {

std::unique_ptr<IMicrobenchmark<size_t>> sharedMemoryReadTest() {
  return std::make_unique<SharedMemoryReadImpl>();
}

std::unique_ptr<IMicrobenchmark<size_t>> sharedMemoryWriteTest() {
  return std::make_unique<SharedMemoryWriteImpl>();
}

} // namespace bandwidth

//---------------------------
//--- constantMemoryTest  ---
//---------------------------

namespace {

constexpr int kConstantFloats = 16384;

__constant__ float g_constantData[kConstantFloats];

//// The methodology draws on SHOC(because on SHOC hasn't constant part)
// Reads 'kUnroll' floats from constant memory per loop step and accumulates them
__global__ void ConstantMemoryReadKernel(float *output, size_t repeats) {
  float sum = 0.0f;
  int index = 0;
  for (size_t j = 0; j < repeats; j++) {
    #pragma unroll
    for (size_t k = 0; k < kUnroll; k++) {
      sum += g_constantData[(index +  k) & (kConstantFloats - 1)];
    }
    index = (index + kUnroll) & (kConstantFloats - 1);
  }

  size_t gid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  output[gid] = sum;
}

// Read
class ConstantMemoryReadImpl : public KernelBandwidthImpl {
  public:
    ConstantMemoryReadImpl() = default;

    virtual std::string Name() const override { return "bandwidth::constantMemoryRead()"; }

    virtual void Init() override {
      KernelBandwidthImpl::Init();

      auto host = std::make_unique<float[]>(kConstantFloats);
      for (int i = 0; i < kConstantFloats; i++) {
        host[i] = static_cast<float>(i);
      }
      HANDLE_ERROR(::cudaMemcpyToSymbol(g_constantData, host.get(),
                                        kConstantFloats * sizeof(float)));
    }

    virtual void SingleRun() override {
      auto repeats = Repeats();
      for (size_t j = 0; j < SubIterations(); j++) {
        ConstantMemoryReadKernel<<<kBlocks, kBlockThreads>>>(output_, repeats);
      }
      HANDLE_ERROR(Api::cudaGetLastError());
    }
};

} // unnamed namespace

namespace bandwidth {

std::unique_ptr<IMicrobenchmark<size_t>> constantMemoryReadTest() {
  return std::make_unique<ConstantMemoryReadImpl>();
}

} // namespace bandwidth