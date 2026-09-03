#include "Tests/Compute.h"

#include "Framework/CudaEventBenchmark.h"

//----------------------------
//--- Half precision setup ---
//----------------------------

// Written with Qwen 3.8: SHOC has no float16 tests at all, so this part has no prototype
// Neither CUDA nor MUSA guarantees that float16 arithmetic is available
#if defined(API_MUSA)
  #if defined(__has_include) && __has_include(<musa_fp16.h>)
    #include <musa_fp16.h>
    #define HAS_FP16 1
  #endif
#elif defined(API_CUDA)
  #if defined(__has_include) && __has_include(<cuda_fp16.h>)
    #include <cuda_fp16.h>
    #define HAS_FP16 1
  #endif
#endif

// Native float16 math requires compute capability 5.3 or higher
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)
  #define NATIVE_FP16 1
#endif


//--------------------
//--- Device types ---
//--------------------

namespace {

// Maps a public type tag onto the real type known by the device compiler
template <typename T> struct DeviceType { using Type = T; };

#if defined(HAS_FP16)
template <> struct DeviceType<compute::half_t>  { using Type = __half;  };
template <> struct DeviceType<compute::half2_t> { using Type = __half2; };
#endif

// Used to build the name of the benchmark, e.g. 'compute::sin<fp32>()'
template <typename T> const char *TypeName();
template <> const char *TypeName<double>()           { return "fp64";   }
template <> const char *TypeName<float>()            { return "fp32";   }
template <> const char *TypeName<compute::half_t>()  { return "fp16";   }
template <> const char *TypeName<compute::half2_t>() { return "fp16x2"; }

// The return type cannot be deduced from the argument, so this one stays a template
template <typename T> __host__ __device__ T MakeValue(float value) {
  return static_cast<T>(value);
}

__device__ float  Add(float a, float b)   { return a + b; }
__device__ float  Sub(float a, float b)   { return a - b; }
__device__ float  Mul(float a, float b)   { return a * b; }
__device__ double Add(double a, double b) { return a + b; }
__device__ double Sub(double a, double b) { return a - b; }
__device__ double Mul(double a, double b) { return a * b; }

// 'sinf()' and 'sin()' are different functions, the second one would silently promote
// a float argument to double and make the fp32 test several times slower
__device__ float  Sin(float x)  { return sinf(x); }
__device__ float  Cos(float x)  { return cosf(x); }
__device__ float  Exp(float x)  { return expf(x); }
__device__ float  Log(float x)  { return logf(x); }

__device__ double Sin(double x) { return sin(x); }
__device__ double Cos(double x) { return cos(x); }
__device__ double Exp(double x) { return exp(x); }
__device__ double Log(double x) { return log(x); }

#if defined(HAS_FP16)

template <> __host__ __device__ __half MakeValue<__half>(float value) {
  return __float2half(value);
}

template <> __host__ __device__ __half2 MakeValue<__half2>(float value) {
  return __float2half2_rn(value);
}

#if defined(NATIVE_FP16)
// '__half' and '__half2' have no usable operators below compute capability 5.3
__device__ __half  Add(__half a, __half b)   { return __hadd(a, b);  }
__device__ __half  Sub(__half a, __half b)   { return __hsub(a, b);  }
__device__ __half  Mul(__half a, __half b)   { return __hmul(a, b);  }
__device__ __half2 Add(__half2 a, __half2 b) { return __hadd2(a, b); }
__device__ __half2 Sub(__half2 a, __half2 b) { return __hsub2(a, b); }
__device__ __half2 Mul(__half2 a, __half2 b) { return __hmul2(a, b); }

__device__ __half  Sin(__half x)  { return hsin(x);  }
__device__ __half  Cos(__half x)  { return hcos(x);  }
__device__ __half  Exp(__half x)  { return hexp(x);  }
__device__ __half  Log(__half x)  { return hlog(x);  }

__device__ __half2 Sin(__half2 x) { return h2sin(x); }
__device__ __half2 Cos(__half2 x) { return h2cos(x); }
__device__ __half2 Exp(__half2 x) { return h2exp(x); }
__device__ __half2 Log(__half2 x) { return h2log(x); }
#else
// Old devices have no native float16 math, so we emulate it via float
__device__ __half Add(__half a, __half b) { return __float2half(__half2float(a) + __half2float(b)); }
__device__ __half Sub(__half a, __half b) { return __float2half(__half2float(a) - __half2float(b)); }
__device__ __half Mul(__half a, __half b) { return __float2half(__half2float(a) * __half2float(b)); }
__device__ __half Sin(__half x) { return __float2half(sinf(__half2float(x))); }
__device__ __half Cos(__half x) { return __float2half(cosf(__half2float(x))); }
__device__ __half Exp(__half x) { return __float2half(expf(__half2float(x))); }
__device__ __half Log(__half x) { return __float2half(logf(__half2float(x))); }

__device__ __half2 Add(__half2 a, __half2 b) { return __halves2half2(Add(__low2half(a), __low2half(b)), Add(__high2half(a), __high2half(b))); }
__device__ __half2 Sub(__half2 a, __half2 b) { return __halves2half2(Sub(__low2half(a), __low2half(b)), Sub(__high2half(a), __high2half(b))); }
__device__ __half2 Mul(__half2 a, __half2 b) { return __halves2half2(Mul(__low2half(a), __low2half(b)), Mul(__high2half(a), __high2half(b))); }
__device__ __half2 Sin(__half2 x) { return __halves2half2(Sin(__low2half(x)), Sin(__high2half(x))); }
__device__ __half2 Cos(__half2 x) { return __halves2half2(Cos(__low2half(x)), Cos(__high2half(x))); }
__device__ __half2 Exp(__half2 x) { return __halves2half2(Exp(__low2half(x)), Exp(__high2half(x))); }
__device__ __half2 Log(__half2 x) { return __halves2half2(Log(__low2half(x)), Log(__high2half(x))); }
#endif

#endif // HAS_FP16

} // unnamed namespace


//-----------------
//--- Operators ---
//-----------------

// SHOC keeps its chains bounded by a linear expression with a fixed point ('s = v1 - s * v2').
// Ours are not linear, so the constants below were found with Qwen 3.8: each seed is the fixed
// point of its own function. A chain that leaves it ends up in a NaN or in the denormals
namespace {

// s = sin(s) + v1, the fixed point of 'sin(s) + 0.5'
struct SinOp {
  static constexpr float kV1 = 0.5f, kInit = 1.4973f;

  template <typename T>
  static __device__ __forceinline__ T Apply(T s, T v1) {
    return Add(Sin(s), v1);
  }
};

// s = cos(s) + v1, the fixed point of 'cos(s) + 0.5'
struct CosOp {
  static constexpr float kV1 = 0.5f, kInit = 1.0218f;

  template <typename T>
  static __device__ __forceinline__ T Apply(T s, T v1) {
    return Add(Cos(s), v1);
  }
};

// s = exp(v1 - s), the fixed point of 'exp(0.5 - s)'
// The subtraction keeps the argument negative, an addition would overflow in a few steps
struct ExpOp {
  static constexpr float kV1 = 0.5f, kInit = 0.7662f;

  template <typename T>
  static __device__ __forceinline__ T Apply(T s, T v1) {
    return Exp(Sub(v1, s));
  }
};

// s = log(s) + v1, the fixed point of 'log(s) + 10.0'
// The shift keeps the argument far from zero, where the next step would get a NaN
struct LogOp {
  static constexpr float kV1 = 10.0f, kInit = 12.5280f;

  template <typename T>
  static __device__ __forceinline__ T Apply(T s, T v1) {
    return Add(Log(s), v1);
  }
};

// The three tests below are the SHOC originals, where the expression is at the same time
// the measured operation and the way to keep the chain bounded

// s = v1 - s, stays at 'v1 / 2'
struct AddOp {
  static constexpr float kV1 = 1.0f, kInit = 0.5f;

  template <typename T>
  static __device__ __forceinline__ T Apply(T s, T v1) {
    return Sub(v1, s);
  }
};

// s = s * v1 with 'v1 = -1', so the value only changes its sign
struct MulOp {
  static constexpr float kV1 = -1.0f, kInit = 1.0f;

  template <typename T>
  static __device__ __forceinline__ T Apply(T s, T v1) {
    return Mul(s, v1);
  }
};

// s = s * v1 + v1, the SHOC expression itself, converges to 'v1 / (1 - v1)'
struct MAddOp {
  static constexpr float kV1 = 0.5f, kInit = 1.0f;

  template <typename T>
  static __device__ __forceinline__ T Apply(T s, T v1) {
    return Add(Mul(s, v1), v1);
  }
};

template <typename T, typename OP> __host__ T V1() { return MakeValue<T>(OP::kV1); }

template <typename T, typename OP> __host__ T InitialValue() { return MakeValue<T>(OP::kInit); }

} // unnamed namespace


//----------------
//--- Geometry ---
//----------------

namespace {

// SHOC uses the same block size
constexpr size_t kBlockThreads = 256;

// Steps performed by a thread per one iteration of the main loop, does not depend on ILP
// SHOC unrolls 240 of them, but its step is one instruction and ours is a couple of dozens
constexpr size_t kStepsPerIteration = 32;

// 'Calibrate()' runs 'kProbeIterations' and scales the result to fit 'kTargetSeconds'
constexpr size_t kProbeIterations = 32;
constexpr double kTargetSeconds   = 0.02;
constexpr size_t kMinIterations   = 16;
constexpr size_t kMaxIterations   = 65536;

constexpr size_t kMaxIlp = 8;

const Api::cudaDeviceProp &Props() {
  static Api::cudaDeviceProp props{};
  static bool loaded = false;

  if (!loaded) {
    int device = 0;
    HANDLE_ERROR(Api::cudaGetDevice(&device));
    HANDLE_ERROR(Api::cudaGetDeviceProperties(&props, device));
    loaded = true;
  }
  return props;
}

// The number of blocks required to fill all the multiprocessors of the GPU
size_t Blocks() {
  const auto &props = Props();
  size_t blocks_per_mp = std::max<size_t>(1, props.maxThreadsPerMultiProcessor / kBlockThreads);
  return std::max<size_t>(1, props.multiProcessorCount) * blocks_per_mp;
}

} // unnamed namespace


//---------------
//--- Kernels ---
//---------------

namespace {

// Every thread keeps 'ILP' independent chains: one chain shows the latency of the function,
// several of them saturate the pipeline and the flat part of the curve is the throughput
template <typename T, int ILP, typename OP>
__global__ void TranscendentalKernel(T *data, uint32_t iterations, T v1) {
  constexpr int UNROLL = (int)kStepsPerIteration / ILP;
  const size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;

  // The seeds come from the memory, so the compiler cannot precompute the chains
  T s[ILP];
  #pragma unroll
  for (int i = 0; i < ILP; i++) {
    s[i] = data[gid * ILP + i];
  }

  // The outer loop is never unrolled, it only repeats the already unrolled body
  #pragma unroll 1
  for (uint32_t j = 0; j < iterations; j++) {
    #pragma unroll
    for (int u = 0; u < UNROLL; u++) {
      #pragma unroll
      for (int i = 0; i < ILP; i++) {
        s[i] = OP::template Apply<T>(s[i], v1);
      }
    }
  }

  // The result must be written back, otherwise the compiler removes the whole loop
  T sum = s[0];
  #pragma unroll
  for (int i = 1; i < ILP; i++) {
    sum = Add(sum, s[i]);
  }
  data[gid * ILP] = sum;
}

template <typename T>
__global__ void FillKernel(T *data, size_t size, T value) {
  size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) {
    data[gid] = value;
  }
}

} // unnamed namespace


//------------------
//--- Validation ---
//------------------

// Our chains converge to a fixed point, so a diverged value means that the constants are wrong
namespace {

template <typename T> bool IsFinite(T value) {
  return std::isfinite(static_cast<double>(value));
}

#if defined(HAS_FP16)

bool IsFinite(__half value) {
  return std::isfinite(__half2float(value));
}

bool IsFinite(__half2 value) {
  // Both lanes run the same chain, so the low one is enough
  // '__low2half()' is a device-only function, hence the raw copy
  __half low{};
  std::memcpy(&low, &value, sizeof(low));
  return IsFinite(low);
}

#endif // HAS_FP16

} // unnamed namespace


//-----------------
//--- Benchmark ---
//-----------------

namespace {

template <typename TAG, typename OP>
class TranscendentalImpl : public CudaEventBenchmark<compute::Ilp> {
    using T = typename DeviceType<TAG>::Type;

  public:
    TranscendentalImpl(std::string name, size_t iterations)
      : name_(std::move(name)), iterations_(iterations) {}

    virtual std::string Name() const override { return name_; }

    virtual void Init() override {
      // The buffer is allocated for the maximal ILP, so all the configurations share its size
      size_t elements = compute::TotalThreads() * kMaxIlp;
      HANDLE_ERROR(Api::cudaMalloc(&data_, elements * sizeof(T)));

      size_t blocks = (elements + kBlockThreads - 1) / kBlockThreads;
      FillKernel<T><<<(unsigned)blocks, (unsigned)kBlockThreads>>>(
        data_, elements, InitialValue<T, OP>());
      HANDLE_ERROR(Api::cudaGetLastError());
    }

    virtual void CleanUp() override {
      // 'CleanUp()' is called after the stop event is recorded, so this check is free
      Validate();
      HANDLE_ERROR(Api::cudaFree(data_));
      data_ = nullptr;
    }

    virtual void SingleRun() override {
      switch (std::get<0>(Args())) {
        case 1: Launch<1>(); break;
        case 2: Launch<2>(); break;
        case 4: Launch<4>(); break;
        case 8: Launch<8>(); break;
        default:
          throw std::runtime_error("Unsupported level of ILP: " +
                                   std::to_string(std::get<0>(Args())));
      }
      HANDLE_ERROR(Api::cudaGetLastError());
    }

  private:
    template <int ILP>
    void Launch() {
      for (size_t j = 0; j < SubIterations(); j++) {
        TranscendentalKernel<T, ILP, OP><<<(unsigned)Blocks(), (unsigned)kBlockThreads>>>(
          data_, (uint32_t)iterations_, V1<T, OP>());
      }
    }

    // The chains of all the threads are identical, so the first value is enough
    void Validate() {
      T value{};
      HANDLE_ERROR(Api::cudaMemcpy(&value, data_, sizeof(T), Api::cudaMemcpyDeviceToHost));
      if (!IsFinite(value)) {
        throw std::runtime_error(name_ + ": the chain of operations has diverged");
      }
    }

    std::string name_;
    size_t iterations_{};
    T *data_{};
};

// Builds a test and gives it a name like 'compute::sin<fp32>()'
template <typename TAG, typename OP>
std::unique_ptr<IMicrobenchmark<compute::Ilp>> MakeTest(const char *op_name, size_t iterations) {
  std::string name = std::string("compute::") + op_name + '<' + TypeName<TAG>() + ">()";
  return std::make_unique<TranscendentalImpl<TAG, OP>>(std::move(name), iterations);
}

} // unnamed namespace


//-------------------
//--- Calibration ---
//-------------------

namespace {

// Measures 'kProbeIterations' and scales them to fit 'kTargetSeconds'
template <typename TAG>
size_t CalibrateImpl() {
  using T = typename DeviceType<TAG>::Type;
  size_t elements = compute::TotalThreads() * kMaxIlp;

  T *data{};
  HANDLE_ERROR(Api::cudaMalloc(&data, elements * sizeof(T)));
  size_t fill_blocks = (elements + kBlockThreads - 1) / kBlockThreads;
  FillKernel<T><<<(unsigned)fill_blocks, (unsigned)kBlockThreads>>>(
    data, elements, InitialValue<T, SinOp>());
  HANDLE_ERROR(Api::cudaGetLastError());

  Api::cudaEvent_t start, stop;
  HANDLE_ERROR(Api::cudaEventCreate(&start));
  HANDLE_ERROR(Api::cudaEventCreate(&stop));
  HANDLE_ERROR(Api::cudaDeviceSynchronize());

  // SHOC probes with 'MulMAdd2', the most expensive of its kernels, and we probe with sin()
  // for the same reason: the remaining tests can only become shorter than the target time.
  // The first launch is a warm-up, its time includes the lazy loading of the module
  float ms{};
  for (int attempt = 0; attempt < 2; attempt++) {
    HANDLE_ERROR(Api::cudaEventRecord(start));
    TranscendentalKernel<T, 2, SinOp><<<(unsigned)Blocks(), (unsigned)kBlockThreads>>>(
      data, (uint32_t)kProbeIterations, V1<T, SinOp>());
    HANDLE_ERROR(Api::cudaEventRecord(stop));
    HANDLE_ERROR(Api::cudaEventSynchronize(stop));
    HANDLE_ERROR(Api::cudaEventElapsedTime(&ms, start, stop));
  }
  HANDLE_ERROR(Api::cudaGetLastError());

  HANDLE_ERROR(Api::cudaEventDestroy(stop));
  HANDLE_ERROR(Api::cudaEventDestroy(start));
  HANDLE_ERROR(Api::cudaFree(data));

  double seconds = ms * 1e-3;
  if (seconds < 1e-9) {
    return kMaxIterations;
  }

  double scaled = kProbeIterations * kTargetSeconds / seconds;
  return std::min(kMaxIterations, std::max(kMinIterations, (size_t)scaled));
}

} // unnamed namespace


//---------------
//--- Exports ---
//---------------

namespace compute {

const std::vector<Ilp> &SupportedIlp() {
  static const std::vector<Ilp> ilp = { 1, 2, 4, 8 };
  return ilp;
}

bool IsFp16Supported() {
#if defined(HAS_FP16)
  // MUSA and HiP report their own architecture versions, so we trust the toolkit
  if (!IsCuda()) {
    return true;
  }
  const auto &props = Props();
  return props.major * 10 + props.minor >= 53;
#else
  return false;
#endif
}

size_t TotalThreads() {
  return Blocks() * kBlockThreads;
}

std::function<double(double, Ilp)> ToCallsPerSecond(size_t calls_per_step, size_t iterations) {
  // The number of steps does not depend on ILP: a thread always performs
  // 'kStepsPerIteration' of them, either as one long chain or as eight short ones
  return [calls_per_step, iterations](double seconds, Ilp) -> double {
    double steps = (double)TotalThreads() * iterations * kStepsPerIteration;
    return seconds > 1e-9 ? steps * calls_per_step / seconds : 0.0;
  };
}

template <typename T> size_t Calibrate() {
  return CalibrateImpl<T>();
}

template <typename T> std::unique_ptr<IMicrobenchmark<Ilp>> sinTest(size_t iterations) {
  return MakeTest<T, SinOp>("sin", iterations);
}

template <typename T> std::unique_ptr<IMicrobenchmark<Ilp>> cosTest(size_t iterations) {
  return MakeTest<T, CosOp>("cos", iterations);
}

template <typename T> std::unique_ptr<IMicrobenchmark<Ilp>> expTest(size_t iterations) {
  return MakeTest<T, ExpOp>("exp", iterations);
}

template <typename T> std::unique_ptr<IMicrobenchmark<Ilp>> logTest(size_t iterations) {
  return MakeTest<T, LogOp>("log", iterations);
}

template <typename T> std::unique_ptr<IMicrobenchmark<Ilp>> addTest(size_t iterations) {
  return MakeTest<T, AddOp>("add", iterations);
}

template <typename T> std::unique_ptr<IMicrobenchmark<Ilp>> mulTest(size_t iterations) {
  return MakeTest<T, MulOp>("mul", iterations);
}

template <typename T> std::unique_ptr<IMicrobenchmark<Ilp>> maddTest(size_t iterations) {
  return MakeTest<T, MAddOp>("madd", iterations);
}

// The host code knows nothing about '__half', so the tests are instantiated here
template size_t Calibrate<double>();
template std::unique_ptr<IMicrobenchmark<Ilp>> sinTest<double>(size_t);
template std::unique_ptr<IMicrobenchmark<Ilp>> cosTest<double>(size_t);
template std::unique_ptr<IMicrobenchmark<Ilp>> expTest<double>(size_t);
template std::unique_ptr<IMicrobenchmark<Ilp>> logTest<double>(size_t);
template std::unique_ptr<IMicrobenchmark<Ilp>> addTest<double>(size_t);
template std::unique_ptr<IMicrobenchmark<Ilp>> mulTest<double>(size_t);
template std::unique_ptr<IMicrobenchmark<Ilp>> maddTest<double>(size_t);

template size_t Calibrate<float>();
template std::unique_ptr<IMicrobenchmark<Ilp>> sinTest<float>(size_t);
template std::unique_ptr<IMicrobenchmark<Ilp>> cosTest<float>(size_t);
template std::unique_ptr<IMicrobenchmark<Ilp>> expTest<float>(size_t);
template std::unique_ptr<IMicrobenchmark<Ilp>> logTest<float>(size_t);
template std::unique_ptr<IMicrobenchmark<Ilp>> addTest<float>(size_t);
template std::unique_ptr<IMicrobenchmark<Ilp>> mulTest<float>(size_t);
template std::unique_ptr<IMicrobenchmark<Ilp>> maddTest<float>(size_t);

#if defined(HAS_FP16)

template size_t Calibrate<half_t>();
template std::unique_ptr<IMicrobenchmark<Ilp>> sinTest<half_t>(size_t);
template std::unique_ptr<IMicrobenchmark<Ilp>> cosTest<half_t>(size_t);
template std::unique_ptr<IMicrobenchmark<Ilp>> expTest<half_t>(size_t);
template std::unique_ptr<IMicrobenchmark<Ilp>> logTest<half_t>(size_t);
template std::unique_ptr<IMicrobenchmark<Ilp>> addTest<half_t>(size_t);
template std::unique_ptr<IMicrobenchmark<Ilp>> mulTest<half_t>(size_t);
template std::unique_ptr<IMicrobenchmark<Ilp>> maddTest<half_t>(size_t);

template size_t Calibrate<half2_t>();
template std::unique_ptr<IMicrobenchmark<Ilp>> sinTest<half2_t>(size_t);
template std::unique_ptr<IMicrobenchmark<Ilp>> cosTest<half2_t>(size_t);
template std::unique_ptr<IMicrobenchmark<Ilp>> expTest<half2_t>(size_t);
template std::unique_ptr<IMicrobenchmark<Ilp>> logTest<half2_t>(size_t);
template std::unique_ptr<IMicrobenchmark<Ilp>> addTest<half2_t>(size_t);
template std::unique_ptr<IMicrobenchmark<Ilp>> mulTest<half2_t>(size_t);
template std::unique_ptr<IMicrobenchmark<Ilp>> maddTest<half2_t>(size_t);

#else

// Without float16 these symbols must still exist, 'IsFp16Supported()' keeps them unused
std::runtime_error NoFp16() {
  return std::runtime_error("float16 is not supported by this toolkit");
}

template <> size_t Calibrate<half_t>()                                    { throw NoFp16(); }
template <> std::unique_ptr<IMicrobenchmark<Ilp>> sinTest<half_t>(size_t) { throw NoFp16(); }
template <> std::unique_ptr<IMicrobenchmark<Ilp>> cosTest<half_t>(size_t) { throw NoFp16(); }
template <> std::unique_ptr<IMicrobenchmark<Ilp>> expTest<half_t>(size_t) { throw NoFp16(); }
template <> std::unique_ptr<IMicrobenchmark<Ilp>> logTest<half_t>(size_t) { throw NoFp16(); }
template <> std::unique_ptr<IMicrobenchmark<Ilp>> addTest<half_t>(size_t) { throw NoFp16(); }
template <> std::unique_ptr<IMicrobenchmark<Ilp>> mulTest<half_t>(size_t) { throw NoFp16(); }
template <> std::unique_ptr<IMicrobenchmark<Ilp>> maddTest<half_t>(size_t) { throw NoFp16(); }

template <> size_t Calibrate<half2_t>()                                    { throw NoFp16(); }
template <> std::unique_ptr<IMicrobenchmark<Ilp>> sinTest<half2_t>(size_t) { throw NoFp16(); }
template <> std::unique_ptr<IMicrobenchmark<Ilp>> cosTest<half2_t>(size_t) { throw NoFp16(); }
template <> std::unique_ptr<IMicrobenchmark<Ilp>> expTest<half2_t>(size_t) { throw NoFp16(); }
template <> std::unique_ptr<IMicrobenchmark<Ilp>> logTest<half2_t>(size_t) { throw NoFp16(); }
template <> std::unique_ptr<IMicrobenchmark<Ilp>> addTest<half2_t>(size_t) { throw NoFp16(); }
template <> std::unique_ptr<IMicrobenchmark<Ilp>> mulTest<half2_t>(size_t) { throw NoFp16(); }
template <> std::unique_ptr<IMicrobenchmark<Ilp>> maddTest<half2_t>(size_t) { throw NoFp16(); }

#endif

} // namespace compute
