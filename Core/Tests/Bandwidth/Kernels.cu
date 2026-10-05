#include "Tests/Bandwidth.h"

#include "Framework/CudaEventBenchmark.h"

//-------------------------------
//--- On-chip memory helpers  ---
//-------------------------------

// The idea is borrowed from the SHOC benchmark (src/cuda/level0/DeviceMemory.cu 516-579):
namespace {

constexpr int     kBlockThreads   = 256;
constexpr int     kUnroll         = 16;

// The number of blocks required to fill all the multiprocessors of the given device
size_t Blocks(const Api::cudaDeviceProp &props) {
  size_t blocks_per_mp = std::max<size_t>(1, props.maxThreadsPerMultiProcessor / kBlockThreads);
  return std::max<size_t>(1, props.multiProcessorCount) * blocks_per_mp;
}

class KernelBandwidthImpl : public CudaEventBenchmark<size_t> {
  public:
    KernelBandwidthImpl() = default;

    virtual void Init() override {
      // The device is asked before every run, 'SingleRun()' reuses the answer
      blocks_ = Blocks(DeviceProperties());
      HANDLE_ERROR(Api::cudaMalloc(&output_, blocks_ * kBlockThreads * sizeof(float)));
    }

    virtual void CleanUp() override {
      HANDLE_ERROR(Api::cudaFree(output_));
      output_ = nullptr;
    }

  protected:
    // Rounded up, so a test never reports more bytes than its kernels have really read or written
    size_t Repeats() const {
      auto block_size = std::get<0>(Args());
      size_t bytes_per_repeat = blocks_ * kBlockThreads * kUnroll * sizeof(float);
      return (block_size + bytes_per_repeat - 1) / bytes_per_repeat;
    }
    size_t blocks_ {};
    float *output_ {};
};

} // unnamed namespace

//------------------------
//--- sharedMemoryTest ---
//------------------------

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
    // A 32-bit conversion, the 64-bit one is slow on some GPUs and would limit the stores
    float value = static_cast<float>(static_cast<unsigned>(j));
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
        SharedMemoryReadKernel<<<(unsigned)blocks_, kBlockThreads>>>(output_, repeats);
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
        SharedMemoryWriteKernel<<<(unsigned)blocks_, kBlockThreads>>>(output_, repeats);
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
        ConstantMemoryReadKernel<<<(unsigned)blocks_, kBlockThreads>>>(output_, repeats);
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