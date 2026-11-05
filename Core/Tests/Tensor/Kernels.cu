#include "Tests/Tensor.h"
#include "Tests/Tensor/Model.h"

#include "Api/Default.h"


//---------------
//--- Support ---
//---------------

#if defined(TENSOR_CORES_ENABLED) && defined(FP16_SUPPORT) && \
    (defined(API_CUDA) || defined(API_MUSA))
  #if __has_include(<mma.h>)
    #include <mma.h>
    #define TENSOR_WMMA_HEADERS
  #endif
#endif

// Unsupported device images, including sm_61, must still compile
// The probe checks the loaded image, so an empty kernel cannot be reported as Tensor performance
#if defined(TENSOR_WMMA_HEADERS) && \
    ((defined(API_CUDA) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700) || \
     (defined(API_MUSA) && defined(__MUSA_ARCH__) && __MUSA_ARCH__ >= 310))
  #define TENSOR_WMMA_DEVICE
#endif

namespace {

namespace model = tensor::model;

} // unnamed namespace


//----------------
//--- Geometry ---
//----------------

namespace {

// The device is queried on every call, because the framework may reset it between tests
Api::cudaDeviceProp Properties() {
  int device{};
  Api::cudaDeviceProp props{};
  HANDLE_ERROR(Api::cudaGetDevice(&device));
  HANDLE_ERROR(Api::cudaGetDeviceProperties(&props, device));
  return props;
}

// The number of blocks required to fill all the multiprocessors of the given device
size_t Blocks(const Api::cudaDeviceProp &props) {
  const size_t per_sm = std::max<size_t>(1, props.maxThreadsPerMultiProcessor / model::kBlockThreads);
  return std::max<size_t>(1, props.multiProcessorCount) * per_sm;
}

} // unnamed namespace


//-----------------
//--- Resources ---
//-----------------

namespace {

// Releases the device memory even if a run fails its validation
template <typename T>
class DeviceBuffer {
  public:
    explicit DeviceBuffer(size_t elements) {
      HANDLE_ERROR(Api::cudaMalloc(&data_, elements * sizeof(T)));
    }
    ~DeviceBuffer() {
      if (data_) {
        (void)Api::cudaFree(data_);
      }
    }

    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;

    T *get() const { return data_; }

  private:
    T *data_{};
};

// Owns one timing event for the duration of a run
class Event {
  public:
    Event() {
      HANDLE_ERROR(Api::cudaEventCreate(&event_));
    }

    ~Event() {
      (void)Api::cudaEventDestroy(event_);
    }

    Event(const Event &) = delete;
    Event &operator=(const Event &) = delete;

    Api::cudaEvent_t get() const { return event_; }

  private:
    Api::cudaEvent_t event_{};
};

} // unnamed namespace


//---------------
//--- Kernels ---
//---------------

namespace {

#if defined(TENSOR_WMMA_HEADERS)

// Reports whether the loaded device image contains the WMMA path
__global__ void CompiledSupportKernel(int *supported) {
  if (threadIdx.x == 0) {
#if defined(TENSOR_WMMA_DEVICE)
    *supported = 1;
#else
    *supported = 0;
#endif
  }
}

// Each warp keeps 'ILP' independent accumulators and performs 32 MMA calls per iteration
// The calls are divided among the chains, so ILP does not change the total work
template <int ILP>
__global__ void WmmaKernel(const __half *a, const __half *b, float *output,
                          uint32_t iterations) {
#if defined(TENSOR_WMMA_DEVICE)
  #if defined(API_CUDA)
    namespace wmma = nvcuda::wmma;
  #else
    namespace wmma = mtmusa::wmma;
  #endif
  wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> af[ILP];
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> bf[ILP];
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> cf[ILP];

  #pragma unroll
  for (int chain = 0; chain < ILP; ++chain) {
    // Different runtime matrices keep the independent chains distinct
    wmma::load_matrix_sync(af[chain], a + chain * model::kTileElements, 16);
    wmma::load_matrix_sync(bf[chain], b + chain * model::kTileElements, 16);
    wmma::fill_fragment(cf[chain], float(chain + 1) * model::kSeedScale);
  }

  #pragma unroll 1
  for (uint32_t iteration = 0; iteration < iterations; ++iteration) {
    #pragma unroll
    for (int step = 0; step < int(model::kMmaPerIteration) / ILP; ++step) {
      #pragma unroll
      for (int chain = 0; chain < ILP; ++chain) {
        wmma::mma_sync(cf[chain], af[chain], bf[chain], cf[chain]);
      }
    }
  }

  const size_t warp = size_t(blockIdx.x) * model::kWarpsPerBlock + threadIdx.x / model::kWarpSize;
  #pragma unroll
  for (int chain = 0; chain < ILP; ++chain) {
    wmma::store_matrix_sync(output + (warp * ILP + chain) * model::kTileElements,
                            cf[chain], 16, wmma::mem_row_major);
  }
#endif
}

#endif // TENSOR_WMMA_HEADERS

} // unnamed namespace


//-----------------
//--- Benchmark ---
//-----------------

namespace {

#if defined(TENSOR_WMMA_HEADERS)

// A run owns its buffers and events, with full-output validation outside the timed interval
class TensorImpl : public IMicrobenchmark<tensor::Ilp, tensor::Iterations> {
  public:
    explicit TensorImpl(tensor::OutputMode mode) : mode_(mode) {}

    virtual std::string Name() const override {
      return mode_ == tensor::OutputMode::Detailed ? DetailedName() : "tensor::wmma<fp16,fp32>()";
    }

    virtual void Configure(size_t samples, size_t launches,
                           tensor::Ilp ilp, tensor::Iterations iterations) override {
      model::CheckConfiguration(ilp, iterations);
      if (samples == 0 || launches == 0) {
        throw std::invalid_argument("Tensor sample counts must be positive");
      }
      ilp_ = ilp;
      samples_ = samples;
      launches_ = launches;
      iterations_ = iterations;
    }

    virtual std::vector<double> Run() override {
      model::CheckConfiguration(ilp_, iterations_);
      if (samples_ == 0 || launches_ == 0) {
        throw std::logic_error("Tensor test is not configured");
      }
      const size_t blocks = Blocks(Properties());
      warps_ = blocks * model::kWarpsPerBlock;
      const size_t input_elements = ilp_ * model::kTileElements;
      const size_t output_elements = warps_ * input_elements;
      std::vector<__half> host_a(input_elements);
      std::vector<__half> host_b(input_elements);
      std::vector<double> expected(input_elements);
      std::vector<float> actual(output_elements);
      for (size_t chain = 0; chain < ilp_; ++chain) {
        for (size_t row = 0; row < model::kTile; ++row) {
          for (size_t col = 0; col < model::kTile; ++col) {
            const size_t offset = chain * model::kTileElements;
            host_a[offset + row * model::kTile + col] = __float2half(model::InputA(chain, row, col));
            // B is loaded as column-major, while the reference uses logical (row, col)
            host_b[offset + col * model::kTile + row] = __float2half(model::InputB(chain, row, col));
            expected[offset + row * model::kTile + col] =
                model::Expected(chain, row, col, ilp_, iterations_);
          }
        }
      }

      DeviceBuffer<__half> a(input_elements);
      DeviceBuffer<__half> b(input_elements);
      DeviceBuffer<float> output(output_elements);
      Event start;
      Event stop;
      HANDLE_ERROR(Api::cudaMemcpy(a.get(), host_a.data(), input_elements * sizeof(__half),
                                   Api::cudaMemcpyHostToDevice));
      HANDLE_ERROR(Api::cudaMemcpy(b.get(), host_b.data(), input_elements * sizeof(__half),
                                   Api::cudaMemcpyHostToDevice));
      std::vector<double> times;
      for (size_t sample = 0; sample < samples_; ++sample) {
        // 0xffffffff is NaN, so a missing store cannot accidentally pass validation
        HANDLE_ERROR(Api::cudaMemset(output.get(), 0xff, output_elements * sizeof(float)));
        HANDLE_ERROR(Api::cudaDeviceSynchronize());
        HANDLE_ERROR(Api::cudaEventRecord(start.get()));
        for (size_t launch = 0; launch < launches_; ++launch) {
          switch (ilp_) {
            case 1: Launch<1>(a.get(), b.get(), output.get(), blocks); break;
            case 2: Launch<2>(a.get(), b.get(), output.get(), blocks); break;
            case 4: Launch<4>(a.get(), b.get(), output.get(), blocks); break;
            case 8: Launch<8>(a.get(), b.get(), output.get(), blocks); break;
          }
        }
        HANDLE_ERROR(Api::cudaGetLastError());
        HANDLE_ERROR(Api::cudaEventRecord(stop.get()));
        HANDLE_ERROR(Api::cudaEventSynchronize(stop.get()));
        float ms{};
        HANDLE_ERROR(Api::cudaEventElapsedTime(&ms, start.get(), stop.get()));
        if (!std::isfinite(ms) || ms <= 0) {
          throw std::runtime_error("Invalid Tensor event duration");
        }
        times.push_back(double(ms) * 1e-3 / launches_);

        // Validation and transfers are outside the timed interval
        // Each launch resets its accumulators, so the reference is not multiplied by 'launches_'
        HANDLE_ERROR(Api::cudaMemcpy(actual.data(), output.get(), output_elements * sizeof(float),
                                     Api::cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < actual.size(); ++i) {
          const double reference = expected[i % input_elements];
          // Binary-fraction inputs and the iteration cap keep this exact in FP32
          if (!std::isfinite(actual[i]) || double(actual[i]) != reference) {
            std::ostringstream message;
            message << DetailedName() << ": validation failed at output " << i
                    << ", expected " << reference << ", got " << actual[i];
            throw std::runtime_error(message.str());
          }
        }
      }
      return times;
    }

  private:
    // The configuration is included in detailed output and in validation errors
    std::string DetailedName() const {
      return "tensor::wmma<fp16,fp32>()[ilp=" + std::to_string(ilp_) +
             ",warps=" + std::to_string(warps_) + ",iterations=" + std::to_string(iterations_) + "]";
    }

    template <int ILP>
    void Launch(const __half *a, const __half *b, float *output, size_t blocks) {
      WmmaKernel<ILP><<<static_cast<unsigned>(blocks), model::kBlockThreads>>>(
          a, b, output, static_cast<uint32_t>(iterations_));
    }

    tensor::OutputMode mode_;
    size_t ilp_{};
    size_t samples_{};
    size_t launches_{};
    size_t iterations_{};
    size_t warps_{};
};

#endif // TENSOR_WMMA_HEADERS

} // unnamed namespace


//---------------
//--- Exports ---
//---------------

namespace tensor {

std::string UnsupportedReason() {
#if !defined(TENSOR_CORES_ENABLED)
  return "disabled at build time (TENSOR_CORES=OFF)";
#elif defined(API_HIP)
  return "the HIP WMMA backend is not implemented";
#elif !defined(TENSOR_WMMA_HEADERS)
  return "this toolkit does not provide the required FP16 and mma.h headers";
#else
  const auto props = Properties();
#if defined(API_CUDA)
  if (props.major < 7) {
    return "CUDA WMMA requires compute capability 7.0 or newer";
  }
  if (std::string(props.name).find("GTX 16") != std::string::npos) {
    return "GeForce GTX 16-series GPUs do not have Tensor Cores";
  }
#elif defined(API_MUSA)
  if (props.major * 10 + props.minor < 31) {
    return "this MUSA WMMA backend targets mp_31 or newer; mp_21/mp_22 are not yet supported";
  }
#endif
  if (props.warpSize != int(model::kWarpSize)) {
    return "this WMMA backend requires 32-thread warps";
  }
  DeviceBuffer<int> flag(1);
  HANDLE_ERROR(Api::cudaMemset(flag.get(), 0, sizeof(int)));
  CompiledSupportKernel<<<1, model::kWarpSize>>>(flag.get());
  HANDLE_ERROR(Api::cudaGetLastError());
  int compiled_support{};
  HANDLE_ERROR(Api::cudaMemcpy(&compiled_support, flag.get(), sizeof(int), Api::cudaMemcpyDeviceToHost));
  if (!compiled_support) {
    return "the loaded device image was compiled without a supported WMMA target";
  }
  return {};
#endif
}

size_t TotalWarps() {
  return Blocks(Properties()) * model::kWarpsPerBlock;
}

std::unique_ptr<IMicrobenchmark<Ilp, Iterations>> MakeTest(OutputMode mode) {
#if defined(TENSOR_WMMA_HEADERS)
  return std::make_unique<TensorImpl>(mode);
#else
  (void)mode;
  throw std::runtime_error("Tensor WMMA is unavailable in this build");
#endif
}

std::function<double(double, Ilp, Iterations)> ToFlops() {
  return [warps = TotalWarps()](double seconds, Ilp, Iterations iterations) {
    if (!std::isfinite(seconds) || seconds <= 0) {
      throw std::runtime_error("Invalid Tensor duration");
    }
    return model::Operations(warps, iterations) / seconds;
  };
}

} // namespace tensor
