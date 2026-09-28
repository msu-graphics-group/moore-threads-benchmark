#include "Tests/Atomics.h"

#include "Framework/CudaEventBenchmark.h"

//-----------------
//--- Operators ---
//-----------------

namespace {

struct AddOp {
  static __device__ __forceinline__ uint32_t Apply(uint32_t *word, uint32_t value, uint32_t) {
    return atomicAdd(word, value);
  }
};

struct ExchOp {
  static __device__ __forceinline__ uint32_t Apply(uint32_t *word, uint32_t value, uint32_t) {
    return atomicExch(word, value);
  }
};

// Even steps swap 0 for the value, odd steps swap it back; failed calls under contention also count
struct CasOp {
  static __device__ __forceinline__ uint32_t Apply(uint32_t *word, uint32_t value, uint32_t step) {
    return step % 2 == 0 ? atomicCAS(word, 0u, value) : atomicCAS(word, value, 0u);
  }
};

} // unnamed namespace


//----------------
//--- Geometry ---
//----------------

namespace {

constexpr size_t kBlockThreads = 256;

constexpr size_t kStepsPerIteration = 16;
static_assert(kStepsPerIteration % 2 == 0, "'CasOp' needs an even number of steps");

// The thread 't' of a block adds, writes or swaps the value 'kFirstValue + t', which is never 0
constexpr uint32_t kFirstValue = 1;

size_t Blocks(const Api::cudaDeviceProp &props) {
  size_t blocks_per_mp = std::max<size_t>(1, props.maxThreadsPerMultiProcessor / kBlockThreads);
  return std::max<size_t>(1, props.multiProcessorCount) * blocks_per_mp;
}

} // unnamed namespace


//---------------
//--- Kernels ---
//---------------

// The idea is borrowed from the Accel-Sim benchmark (gpu-app-collection,
// src/cuda/GPU_Microbenchmark/ubench/atomics/Atomic_add_bw/atomic_add_bw.cu 18-49):
namespace {

// Performs 'iterations * kStepsPerIteration' calls on one word and returns the sum of the old values
template <typename OP>
__device__ __forceinline__ uint32_t Steps(uint32_t *word, uint32_t value, uint32_t iterations) {
  uint32_t sum = 0;

  #pragma unroll 1
  for (uint32_t j = 0; j < iterations; j++) {
    #pragma unroll
    for (uint32_t k = 0; k < kStepsPerIteration; k++) {
      sum += OP::Apply(word, value, k);
    }
  }
  return sum;
}

// 'contention' is a runtime value, otherwise the compiler merges the atomicAdd() calls of a warp
template <typename OP>
__global__ void GlobalAtomicKernel(uint32_t *words, uint32_t *sums, uint32_t contention,
                                   uint32_t iterations) {
  const size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t *word = &words[(size_t)blockIdx.x * blockDim.x + threadIdx.x / contention];

  // The sum must be stored, otherwise atomicAdd() becomes a RED instead of an ATOM
  sums[gid] = Steps<OP>(word, kFirstValue + threadIdx.x, iterations);
}

template <typename OP>
__global__ void SharedAtomicKernel(uint32_t *words, uint32_t *sums, uint32_t contention,
                                   uint32_t iterations) {
  __shared__ uint32_t block_words[kBlockThreads];
  const size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

  block_words[threadIdx.x] = 0;
  __syncthreads();

  sums[gid] = Steps<OP>(&block_words[threadIdx.x / contention], kFirstValue + threadIdx.x, iterations);

  __syncthreads();
  words[gid] = block_words[threadIdx.x];
}

} // unnamed namespace


//------------------
//--- Validation ---
//------------------

namespace {

struct BlockState {
  std::vector<uint32_t> words;
  std::vector<uint32_t> sums;
  size_t contention{};
  uint32_t steps{};
  uint32_t launches{};
};

uint32_t ValueOf(size_t t) {
  return kFirstValue + static_cast<uint32_t>(t);
}

bool IsValid(AddOp, const BlockState &state) {
  for (size_t w = 0; w < kBlockThreads / state.contention; w++) {
    uint32_t expected = 0;
    for (size_t t = w * state.contention; t < (w + 1) * state.contention; t++) {
      expected += state.launches * state.steps * ValueOf(t);
    }
    if (state.words[w] != expected) {
      return false;
    }
  }
  return true;
}

bool IsValid(ExchOp, const BlockState &state) {
  for (size_t w = 0; w < kBlockThreads / state.contention; w++) {
    bool found = false;
    uint32_t returned = state.words[w], written = 0;
    for (size_t t = w * state.contention; t < (w + 1) * state.contention; t++) {
      found = found || state.words[w] == ValueOf(t);
      returned += state.sums[t];
      written += state.steps * ValueOf(t);
    }
    if (!found || (state.launches == 1 && returned != written)) {
      return false;
    }
  }

  if (state.contention == 1) {
    uint32_t own_values = state.launches == 1 ? state.steps - 1 : state.steps;
    for (size_t t = 0; t < kBlockThreads; t++) {
      if (state.sums[t] != own_values * ValueOf(t)) {
        return false;
      }
    }
  }
  return true;
}

// Every word returns to 0 even without atomicity, so this check cannot catch a lost update
bool IsValid(CasOp, const BlockState &state) {
  for (size_t w = 0; w < kBlockThreads / state.contention; w++) {
    if (state.words[w] != 0) {
      return false;
    }
  }

  if (state.contention == 1) {
    for (size_t t = 0; t < kBlockThreads; t++) {
      if (state.sums[t] != state.steps / 2 * ValueOf(t)) {
        return false;
      }
    }
  }
  return true;
}

} // unnamed namespace


//-----------------
//--- Benchmark ---
//-----------------

namespace {

enum class Space { Global, Shared };

template <Space SPACE, typename OP>
class AtomicImpl : public CudaEventBenchmark<atomics::Contention, atomics::Iterations> {
  public:
    explicit AtomicImpl(std::string name) : name_(std::move(name)) {}

    virtual std::string Name() const override { return name_; }

    virtual void Init() override {
      blocks_ = Blocks(DeviceProperties());

      size_t elements = blocks_ * kBlockThreads;
      HANDLE_ERROR(Api::cudaMalloc(&words_, elements * sizeof(uint32_t)));
      HANDLE_ERROR(Api::cudaMalloc(&sums_, elements * sizeof(uint32_t)));
      HANDLE_ERROR(Api::cudaMemset(words_, 0, elements * sizeof(uint32_t)));
    }

    virtual void CleanUp() override {
      Validate();
      HANDLE_ERROR(Api::cudaFree(sums_));
      HANDLE_ERROR(Api::cudaFree(words_));
      sums_ = nullptr;
      words_ = nullptr;
    }

    virtual void SingleRun() override {
      auto contention = std::get<0>(Args());
      if (contention == 0 || kBlockThreads % contention != 0) {
        throw std::runtime_error("Unsupported level of contention: " + std::to_string(contention));
      }

      auto iterations = static_cast<uint32_t>(std::get<1>(Args()));
      for (size_t j = 0; j < SubIterations(); j++) {
        if constexpr (SPACE == Space::Global) {
          GlobalAtomicKernel<OP><<<(unsigned)blocks_, (unsigned)kBlockThreads>>>(
            words_, sums_, (uint32_t)contention, iterations);
        } else {
          SharedAtomicKernel<OP><<<(unsigned)blocks_, (unsigned)kBlockThreads>>>(
            words_, sums_, (uint32_t)contention, iterations);
        }
      }
      HANDLE_ERROR(Api::cudaGetLastError());
    }

  private:
    void Validate() {
      BlockState state;
      state.words.resize(kBlockThreads);
      state.sums.resize(kBlockThreads);
      HANDLE_ERROR(Api::cudaMemcpy(state.words.data(), words_, kBlockThreads * sizeof(uint32_t),
                                   Api::cudaMemcpyDeviceToHost));
      HANDLE_ERROR(Api::cudaMemcpy(state.sums.data(), sums_, kBlockThreads * sizeof(uint32_t),
                                   Api::cudaMemcpyDeviceToHost));
      state.contention = std::get<0>(Args());
      state.steps = static_cast<uint32_t>(std::get<1>(Args()) * kStepsPerIteration);

      // The global words are set to 0 only in 'Init()', the shared ones in every launch
      state.launches = SPACE == Space::Global ? static_cast<uint32_t>(SubIterations()) : 1;

      bool untouched = std::all_of(state.words.begin() + kBlockThreads / state.contention,
                                   state.words.end(), [](uint32_t word) { return word == 0; });
      if (!untouched || !IsValid(OP{}, state)) {
        throw std::runtime_error(name_ + ": the final values are wrong, the atomic operations failed");
      }
    }

    std::string name_;
    size_t blocks_{};
    uint32_t *words_{};
    uint32_t *sums_{};
};

} // unnamed namespace


//---------------
//--- Exports ---
//---------------

namespace atomics {

// The levels of contention follow Jia et al., 'Dissecting the NVidia Turing T4 GPU
// via Microbenchmarking' (2019), section 4.2
const std::vector<Contention> &SupportedContention() {
  static const std::vector<Contention> contention = { 1, kBlockThreads };
  return contention;
}

size_t TotalThreads() {
  int device{};
  Api::cudaDeviceProp props{};
  HANDLE_ERROR(Api::cudaGetDevice(&device));
  HANDLE_ERROR(Api::cudaGetDeviceProperties(&props, device));
  return Blocks(props) * kBlockThreads;
}

std::function<double(double, Contention, Iterations)> ToCallsPerSecond() {
  return [](double seconds, Contention, Iterations iterations) -> double {
    double calls = (double)TotalThreads() * iterations * kStepsPerIteration;
    return seconds > 1e-9 ? calls / seconds : 0.0;
  };
}

std::unique_ptr<IMicrobenchmark<Contention, Iterations>> globalAtomicAddTest() {
  return std::make_unique<AtomicImpl<Space::Global, AddOp>>("atomics::atomicAdd<uint32>()::global");
}

std::unique_ptr<IMicrobenchmark<Contention, Iterations>> sharedAtomicAddTest() {
  return std::make_unique<AtomicImpl<Space::Shared, AddOp>>("atomics::atomicAdd<uint32>()::shared");
}

std::unique_ptr<IMicrobenchmark<Contention, Iterations>> globalAtomicExchTest() {
  return std::make_unique<AtomicImpl<Space::Global, ExchOp>>("atomics::atomicExch<uint32>()::global");
}

std::unique_ptr<IMicrobenchmark<Contention, Iterations>> sharedAtomicExchTest() {
  return std::make_unique<AtomicImpl<Space::Shared, ExchOp>>("atomics::atomicExch<uint32>()::shared");
}

std::unique_ptr<IMicrobenchmark<Contention, Iterations>> globalAtomicCasTest() {
  return std::make_unique<AtomicImpl<Space::Global, CasOp>>("atomics::atomicCAS<uint32>()::global");
}

std::unique_ptr<IMicrobenchmark<Contention, Iterations>> sharedAtomicCasTest() {
  return std::make_unique<AtomicImpl<Space::Shared, CasOp>>("atomics::atomicCAS<uint32>()::shared");
}

} // namespace atomics
