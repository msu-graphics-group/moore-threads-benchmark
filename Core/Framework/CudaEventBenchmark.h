#pragma once
#include "Defs.h"

#include "Api/Default.h"
#include "IMicrobenchmark.h"


// Basic class that implements 'IMicrobenchmark' using CUDA events to measure time asynchronously
template<typename ...ARGS>
class CudaEventBenchmark : public IMicrobenchmark<ARGS...> {
  public:
    // The member of 'IMicrobenchmark'
    virtual void Configure(size_t iterations, size_t sub_iterations, ARGS... args) override final {
      iterations_ = iterations;
      sub_iterations_ = sub_iterations;
      args_ = std::make_tuple(args...);
    }

    // The member of 'IMicrobenchmark'
    virtual std::vector<double> Run() override final {
      Api::cudaEvent_t start_event, stop_event;
      HANDLE_ERROR(Api::cudaEventCreate(&start_event));
      HANDLE_ERROR(Api::cudaEventCreate(&stop_event));
      HANDLE_ERROR(Api::cudaDeviceSynchronize());

      std::vector<double> times;
      times.reserve(Iterations());
      for (size_t i = 0; i < Iterations(); i++) {
        Init();
        HANDLE_ERROR(Api::cudaDeviceSynchronize());
        HANDLE_ERROR(Api::cudaEventRecord(start_event));
        SingleRun();
        HANDLE_ERROR(Api::cudaEventRecord(stop_event));
        CleanUp();

        float ms{};
        HANDLE_ERROR(Api::cudaEventSynchronize(stop_event));
        HANDLE_ERROR(Api::cudaEventElapsedTime(&ms, start_event, stop_event));
        times.emplace_back(ms * 1e-3 / SubIterations());
      }

      HANDLE_ERROR(Api::cudaEventDestroy(stop_event));
      HANDLE_ERROR(Api::cudaEventDestroy(start_event));

      return times;
    }

  protected:
    CudaEventBenchmark() = default;

    // Initializes the benchmark, allocates the required resources
    virtual void Init() {}

    // Just performs a series of inner subiterations
    // The elapse time will be measured by 'Run()'
    virtual void SingleRun() = 0;

    // Cleans up any resources used by the benchmark
    virtual void CleanUp() {}

    // Returns the value passed to 'Configure()'
    size_t Iterations() const { return iterations_; }

    // Returns the value passed to 'Configure()'
    size_t SubIterations() const { return sub_iterations_; }
    
    // Returns the value passed to 'Configure()'
    std::tuple<ARGS...> Args() const { return args_; }

  private:
    size_t iterations_{};
    size_t sub_iterations_{};
    std::tuple<ARGS...> args_{};
};
