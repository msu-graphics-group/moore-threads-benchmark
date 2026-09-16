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
    #define HAS_FP16
  #endif
#elif defined(API_CUDA)
  #if defined(__has_include) && __has_include(<cuda_fp16.h>)
    #include <cuda_fp16.h>
    #define HAS_FP16
  #endif
#endif


//------------------
//--- Math setup ---
//------------------

// sin(), cos(), exp() and log() use the special function unit by default,
// replace '1' with '0' to measure the library versions instead
#if 1
  #define SFU_MATH
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
template <typename T> std::string_view TypeName();
template <> std::string_view TypeName<double>()           { return "fp64";   }
template <> std::string_view TypeName<float>()            { return "fp32";   }
template <> std::string_view TypeName<compute::half_t>()  { return "fp16";   }
template <> std::string_view TypeName<compute::half2_t>() { return "fp16x2"; }
template <> std::string_view TypeName<int32_t>()          { return "int32";  }
template <> std::string_view TypeName<uint32_t>()         { return "uint32"; }

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

// Signed overflow is undefined, so int32 wraps through uint32 just like the hardware does
__device__ uint32_t Add(uint32_t a, uint32_t b) { return a + b; }
__device__ uint32_t Sub(uint32_t a, uint32_t b) { return a - b; }
__device__ uint32_t Mul(uint32_t a, uint32_t b) { return a * b; }
__device__ int32_t  Add(int32_t a, int32_t b)   { return (int32_t)Add((uint32_t)a, (uint32_t)b); }
__device__ int32_t  Sub(int32_t a, int32_t b)   { return (int32_t)Sub((uint32_t)a, (uint32_t)b); }
__device__ int32_t  Mul(int32_t a, int32_t b)   { return (int32_t)Mul((uint32_t)a, (uint32_t)b); }

// The idea was proposed by Qwen 3.8: every integer step is XORed with 'v1', and this XOR is counted as
// one more IntOp (see 'AddArithmeticTests()' in Main.cpp); the code below implements this idea
// Integer arithmetic is exact, so 'v1 - (v1 - s)' is folded into 's' and the whole chain
// disappears; a XOR with the runtime 'v1' has no such identity and keeps every step alive
template <typename T> __device__ __forceinline__ T Mix(T value, T v1) {
  if constexpr (std::is_integral_v<T>) {
    return value ^ v1;
  } else {
    return value;
  }
}

#if defined(SFU_MATH)
// Only float32 has intrinsics for these functions, they compile directly into the special function unit
__device__ float  Sin(float x)  { return __sinf(x); }
__device__ float  Cos(float x)  { return __cosf(x); }
__device__ float  Exp(float x)  { return __expf(x); }
__device__ float  Log(float x)  { return __logf(x); }
#else
// 'sinf()' and 'sin()' are different functions, the second one would silently promote
// a float argument to double and make the fp32 test several times slower
__device__ float  Sin(float x)  { return sinf(x); }
__device__ float  Cos(float x)  { return cosf(x); }
__device__ float  Exp(float x)  { return expf(x); }
__device__ float  Log(float x)  { return logf(x); }

// float64 has no intrinsics for these functions, so it is measured only with the library
__device__ double Sin(double x) { return sin(x); }
__device__ double Cos(double x) { return cos(x); }
__device__ double Exp(double x) { return exp(x); }
__device__ double Log(double x) { return log(x); }
#endif

#if defined(HAS_FP16)

template <> __host__ __device__ __half MakeValue<__half>(float value) {
  return __float2half(value);
}

template <> __host__ __device__ __half2 MakeValue<__half2>(float value) {
  return __float2half2_rn(value);
}

__device__ __half  Add(__half a, __half b)   { return __hadd(a, b);  }
__device__ __half  Sub(__half a, __half b)   { return __hsub(a, b);  }
__device__ __half  Mul(__half a, __half b)   { return __hmul(a, b);  }
__device__ __half2 Add(__half2 a, __half2 b) { return __hadd2(a, b); }
__device__ __half2 Sub(__half2 a, __half2 b) { return __hsub2(a, b); }
__device__ __half2 Mul(__half2 a, __half2 b) { return __hmul2(a, b); }

#if !defined(SFU_MATH)
// float16 has no intrinsics for these functions either, 'hsin()' and others are the library versions
__device__ __half  Sin(__half x)  { return hsin(x);  }
__device__ __half  Cos(__half x)  { return hcos(x);  }
__device__ __half  Exp(__half x)  { return hexp(x);  }
__device__ __half  Log(__half x)  { return hlog(x);  }

__device__ __half2 Sin(__half2 x) { return h2sin(x); }
__device__ __half2 Cos(__half2 x) { return h2cos(x); }
__device__ __half2 Exp(__half2 x) { return h2exp(x); }
__device__ __half2 Log(__half2 x) { return h2log(x); }
#endif

#endif // HAS_FP16

} // unnamed namespace


//-----------------
//--- Operators ---
//-----------------

// SHOC keeps its chains bounded by a linear expression with a fixed point ('s = v1 - s * v2').
// Ours are not linear, so the constants below were found with Qwen 3.8: each seed is the fixed
// point of its own function. A chain that leaves it ends up in a NaN or in the denormals
// Integer chains need no fixed point: they wrap around, and 'Mix()' keeps them from being folded
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

// The three tests below follow the SHOC arithmetic tests: for floats the expression is at the same time
// the measured operation and the way to keep the chain bounded

// s = v1 - s, a float chain stays at 'v1 / 2'
struct AddOp {
  static constexpr float kV1 = 1.0f, kInit = 0.5f;
  static constexpr int   kIntV1 = 0x5bd1e995, kIntInit = 12345;

  template <typename T>
  static __device__ __forceinline__ T Apply(T s, T v1) {
    return Mix(Sub(v1, s), v1);
  }
};

// s = s * v1, with 'v1 = -1' a float chain only changes its sign
struct MulOp {
  static constexpr float kV1 = -1.0f, kInit = 1.0f;
  static constexpr int   kIntV1 = 0x5bd1e995, kIntInit = 12345;

  template <typename T>
  static __device__ __forceinline__ T Apply(T s, T v1) {
    return Mix(Mul(s, v1), v1);
  }
};

// s = s * v1 + v1, a float chain converges to 'v1 / (1 - v1)'
struct MAddOp {
  static constexpr float kV1 = 0.5f, kInit = 1.0f;
  static constexpr int   kIntV1 = 0x5bd1e995, kIntInit = 12345;

  template <typename T>
  static __device__ __forceinline__ T Apply(T s, T v1) {
    return Mix(Add(Mul(s, v1), v1), v1);
  }
};

// Integers need their own constants, the float ones would be truncated
// 0x5bd1e995 is odd (a MurmurHash2 constant), so multiplying by it loses no bits, and the seed
// is not 0, because 0 never leaves the integer add and madd chains
template <typename T, typename OP> __host__ T V1() {
  if constexpr (std::is_integral_v<T>) {
    return static_cast<T>(OP::kIntV1);
  } else {
    return MakeValue<T>(OP::kV1);
  }
}

template <typename T, typename OP> __host__ T InitialValue() {
  if constexpr (std::is_integral_v<T>) {
    return static_cast<T>(OP::kIntInit);
  } else {
    return MakeValue<T>(OP::kInit);
  }
}

} // unnamed namespace


//----------------
//--- Geometry ---
//----------------

namespace {

// SHOC uses the same block size
constexpr size_t kBlockThreads = 256;

// Steps performed by a thread per one iteration of the main loop, does not depend on ILP
// SHOC unrolls 240 of them, we unroll fewer to keep the body short in the library build, where a sin()
// step is much longer than in SHOC; 32 is also divisible by every supported level of ILP
constexpr size_t kStepsPerIteration = 32;

constexpr size_t kMaxIlp = 8;

// The number of blocks required to fill all the multiprocessors of the given device
size_t Blocks(const Api::cudaDeviceProp &props) {
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

// The float chains stay bounded, so a diverged value means that the constants are wrong
// Integer chains cannot diverge, for them the check always passes
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
class TranscendentalImpl : public CudaEventBenchmark<compute::Ilp, compute::Iterations> {
    using T = typename DeviceType<TAG>::Type;

  public:
    explicit TranscendentalImpl(std::string name) : name_(std::move(name)) {}

    virtual std::string Name() const override { return name_; }

    virtual void Init() override {
      // The device is asked before every run, 'SingleRun()' reuses the answer
      blocks_ = Blocks(DeviceProperties());

      // The buffer is allocated for the maximal ILP, so all the configurations share its size
      size_t elements = blocks_ * kBlockThreads * kMaxIlp;
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
      // The second argument of the configuration, see 'compute::Iterations'
      auto iterations = static_cast<uint32_t>(std::get<1>(Args()));
      for (size_t j = 0; j < SubIterations(); j++) {
        TranscendentalKernel<T, ILP, OP><<<(unsigned)blocks_, (unsigned)kBlockThreads>>>(
          data_, iterations, V1<T, OP>());
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
    size_t blocks_{};
    T *data_{};
};

// Builds a test and gives it a name like 'compute::sin<fp32>()'
template <typename TAG, typename OP>
std::unique_ptr<IMicrobenchmark<compute::Ilp, compute::Iterations>> MakeTest(std::string_view op_name) {
  // C++17 has no 'operator+' for 'std::string' and 'std::string_view', so the name is appended
  std::string name{ "compute::" };
  name += op_name;
  name += '<';
  name += TypeName<TAG>();
  name += ">()";
  return std::make_unique<TranscendentalImpl<TAG, OP>>(std::move(name));
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

// Every supported GPU has native float16, so only the toolkit matters
bool IsFp16Supported() {
#if defined(HAS_FP16)
  return true;
#else
  return false;
#endif
}

// With the special function unit only float32 is supported, it is the only type with intrinsics
template <typename T> bool IsMathSupported() {
#if defined(SFU_MATH)
  return std::is_same_v<T, float>;
#else
  return true;
#endif
}

size_t TotalThreads() {
  // The device is queried on every call, nothing is cached
  int device = 0;
  Api::cudaDeviceProp props{};
  HANDLE_ERROR(Api::cudaGetDevice(&device));
  HANDLE_ERROR(Api::cudaGetDeviceProperties(&props, device));
  return Blocks(props) * kBlockThreads;
}

std::function<double(double, Ilp, Iterations)> ToCallsPerSecond(size_t calls_per_step) {
  // The number of steps does not depend on ILP: a thread always performs
  // 'kStepsPerIteration' of them, either as one long chain or as eight short ones
  return [calls_per_step](double seconds, Ilp, Iterations iterations) -> double {
    double steps = (double)TotalThreads() * iterations * kStepsPerIteration;
    return seconds > 1e-9 ? steps * calls_per_step / seconds : 0.0;
  };
}

template <typename T> std::unique_ptr<IMicrobenchmark<Ilp, Iterations>> sinTest() {
  return MakeTest<T, SinOp>("sin");
}

template <typename T> std::unique_ptr<IMicrobenchmark<Ilp, Iterations>> cosTest() {
  return MakeTest<T, CosOp>("cos");
}

template <typename T> std::unique_ptr<IMicrobenchmark<Ilp, Iterations>> expTest() {
  return MakeTest<T, ExpOp>("exp");
}

template <typename T> std::unique_ptr<IMicrobenchmark<Ilp, Iterations>> logTest() {
  return MakeTest<T, LogOp>("log");
}

template <typename T> std::unique_ptr<IMicrobenchmark<Ilp, Iterations>> addTest() {
  return MakeTest<T, AddOp>("add");
}

template <typename T> std::unique_ptr<IMicrobenchmark<Ilp, Iterations>> mulTest() {
  return MakeTest<T, MulOp>("mul");
}

template <typename T> std::unique_ptr<IMicrobenchmark<Ilp, Iterations>> maddTest() {
  return MakeTest<T, MAddOp>("madd");
}

// The host code knows nothing about '__half', so the tests are instantiated here
template bool IsMathSupported<double>();
template bool IsMathSupported<float>();
template bool IsMathSupported<half_t>();
template bool IsMathSupported<half2_t>();

template std::unique_ptr<IMicrobenchmark<Ilp, Iterations>> sinTest<float>();
template std::unique_ptr<IMicrobenchmark<Ilp, Iterations>> cosTest<float>();
template std::unique_ptr<IMicrobenchmark<Ilp, Iterations>> expTest<float>();
template std::unique_ptr<IMicrobenchmark<Ilp, Iterations>> logTest<float>();

template std::unique_ptr<IMicrobenchmark<Ilp, Iterations>> addTest<double>();
template std::unique_ptr<IMicrobenchmark<Ilp, Iterations>> mulTest<double>();
template std::unique_ptr<IMicrobenchmark<Ilp, Iterations>> maddTest<double>();

template std::unique_ptr<IMicrobenchmark<Ilp, Iterations>> addTest<float>();
template std::unique_ptr<IMicrobenchmark<Ilp, Iterations>> mulTest<float>();
template std::unique_ptr<IMicrobenchmark<Ilp, Iterations>> maddTest<float>();

template std::unique_ptr<IMicrobenchmark<Ilp, Iterations>> addTest<int32_t>();
template std::unique_ptr<IMicrobenchmark<Ilp, Iterations>> mulTest<int32_t>();
template std::unique_ptr<IMicrobenchmark<Ilp, Iterations>> maddTest<int32_t>();

template std::unique_ptr<IMicrobenchmark<Ilp, Iterations>> addTest<uint32_t>();
template std::unique_ptr<IMicrobenchmark<Ilp, Iterations>> mulTest<uint32_t>();
template std::unique_ptr<IMicrobenchmark<Ilp, Iterations>> maddTest<uint32_t>();

#if defined(HAS_FP16)
template std::unique_ptr<IMicrobenchmark<Ilp, Iterations>> addTest<half_t>();
template std::unique_ptr<IMicrobenchmark<Ilp, Iterations>> mulTest<half_t>();
template std::unique_ptr<IMicrobenchmark<Ilp, Iterations>> maddTest<half_t>();

template std::unique_ptr<IMicrobenchmark<Ilp, Iterations>> addTest<half2_t>();
template std::unique_ptr<IMicrobenchmark<Ilp, Iterations>> mulTest<half2_t>();
template std::unique_ptr<IMicrobenchmark<Ilp, Iterations>> maddTest<half2_t>();
#endif

#if defined(SFU_MATH)

// Without the library math these symbols must still exist, 'IsMathSupported()' keeps them unused
std::runtime_error NoIntrinsics() {
  return std::runtime_error("only float32 has intrinsics for sin(), cos(), exp() and log()");
}

template <> std::unique_ptr<IMicrobenchmark<Ilp, Iterations>> sinTest<double>() { throw NoIntrinsics(); }
template <> std::unique_ptr<IMicrobenchmark<Ilp, Iterations>> cosTest<double>() { throw NoIntrinsics(); }
template <> std::unique_ptr<IMicrobenchmark<Ilp, Iterations>> expTest<double>() { throw NoIntrinsics(); }
template <> std::unique_ptr<IMicrobenchmark<Ilp, Iterations>> logTest<double>() { throw NoIntrinsics(); }

#if defined(HAS_FP16)
template <> std::unique_ptr<IMicrobenchmark<Ilp, Iterations>> sinTest<half_t>() { throw NoIntrinsics(); }
template <> std::unique_ptr<IMicrobenchmark<Ilp, Iterations>> cosTest<half_t>() { throw NoIntrinsics(); }
template <> std::unique_ptr<IMicrobenchmark<Ilp, Iterations>> expTest<half_t>() { throw NoIntrinsics(); }
template <> std::unique_ptr<IMicrobenchmark<Ilp, Iterations>> logTest<half_t>() { throw NoIntrinsics(); }

template <> std::unique_ptr<IMicrobenchmark<Ilp, Iterations>> sinTest<half2_t>() { throw NoIntrinsics(); }
template <> std::unique_ptr<IMicrobenchmark<Ilp, Iterations>> cosTest<half2_t>() { throw NoIntrinsics(); }
template <> std::unique_ptr<IMicrobenchmark<Ilp, Iterations>> expTest<half2_t>() { throw NoIntrinsics(); }
template <> std::unique_ptr<IMicrobenchmark<Ilp, Iterations>> logTest<half2_t>() { throw NoIntrinsics(); }
#endif

#else

template std::unique_ptr<IMicrobenchmark<Ilp, Iterations>> sinTest<double>();
template std::unique_ptr<IMicrobenchmark<Ilp, Iterations>> cosTest<double>();
template std::unique_ptr<IMicrobenchmark<Ilp, Iterations>> expTest<double>();
template std::unique_ptr<IMicrobenchmark<Ilp, Iterations>> logTest<double>();

#if defined(HAS_FP16)
template std::unique_ptr<IMicrobenchmark<Ilp, Iterations>> sinTest<half_t>();
template std::unique_ptr<IMicrobenchmark<Ilp, Iterations>> cosTest<half_t>();
template std::unique_ptr<IMicrobenchmark<Ilp, Iterations>> expTest<half_t>();
template std::unique_ptr<IMicrobenchmark<Ilp, Iterations>> logTest<half_t>();

template std::unique_ptr<IMicrobenchmark<Ilp, Iterations>> sinTest<half2_t>();
template std::unique_ptr<IMicrobenchmark<Ilp, Iterations>> cosTest<half2_t>();
template std::unique_ptr<IMicrobenchmark<Ilp, Iterations>> expTest<half2_t>();
template std::unique_ptr<IMicrobenchmark<Ilp, Iterations>> logTest<half2_t>();
#endif

#endif

#if !defined(HAS_FP16)

// Without float16 these symbols must still exist, 'IsFp16Supported()' keeps them unused
std::runtime_error NoFp16() {
  return std::runtime_error("float16 is not supported by this toolkit");
}

template <> std::unique_ptr<IMicrobenchmark<Ilp, Iterations>> sinTest<half_t>()  { throw NoFp16(); }
template <> std::unique_ptr<IMicrobenchmark<Ilp, Iterations>> cosTest<half_t>()  { throw NoFp16(); }
template <> std::unique_ptr<IMicrobenchmark<Ilp, Iterations>> expTest<half_t>()  { throw NoFp16(); }
template <> std::unique_ptr<IMicrobenchmark<Ilp, Iterations>> logTest<half_t>()  { throw NoFp16(); }
template <> std::unique_ptr<IMicrobenchmark<Ilp, Iterations>> addTest<half_t>()  { throw NoFp16(); }
template <> std::unique_ptr<IMicrobenchmark<Ilp, Iterations>> mulTest<half_t>()  { throw NoFp16(); }
template <> std::unique_ptr<IMicrobenchmark<Ilp, Iterations>> maddTest<half_t>() { throw NoFp16(); }

template <> std::unique_ptr<IMicrobenchmark<Ilp, Iterations>> sinTest<half2_t>()  { throw NoFp16(); }
template <> std::unique_ptr<IMicrobenchmark<Ilp, Iterations>> cosTest<half2_t>()  { throw NoFp16(); }
template <> std::unique_ptr<IMicrobenchmark<Ilp, Iterations>> expTest<half2_t>()  { throw NoFp16(); }
template <> std::unique_ptr<IMicrobenchmark<Ilp, Iterations>> logTest<half2_t>()  { throw NoFp16(); }
template <> std::unique_ptr<IMicrobenchmark<Ilp, Iterations>> addTest<half2_t>()  { throw NoFp16(); }
template <> std::unique_ptr<IMicrobenchmark<Ilp, Iterations>> mulTest<half2_t>()  { throw NoFp16(); }
template <> std::unique_ptr<IMicrobenchmark<Ilp, Iterations>> maddTest<half2_t>() { throw NoFp16(); }

#endif

} // namespace compute
