#pragma once
#include "Defs.h"

// Just a classical microbenchmark
// Some external framework will call it and analyze its results
template<typename ...ARGS>
class IMicrobenchmark {
  public:
    IMicrobenchmark(const IMicrobenchmark &) = delete;
    IMicrobenchmark &operator=(const IMicrobenchmark &) = delete;
    virtual ~IMicrobenchmark() = default;

    // The name of the benchmark
    // E.g., 'Overheads::cudaMalloc()'
    virtual std::string Name() const = 0;

    // Each iteration is divided into several sub-iterations
    // This allows the benchmark to reduce overheads by performing multiple operations
    // without synchronization or timing measurements
    virtual void Configure(size_t iterations, size_t sub_iterations, ARGS... args) = 0;

    // Initializes the benchmark, allocates the required resources
    virtual void Init() {}

    // Runs the benchmark with the given arguments 'iterations * sub_iterations' times
    // Returns times elapsed by each outer iteration, in seconds
    // If some iterations are reserved for warm-up and outliers, they must be taken into account by the caller
    virtual std::vector<double> Run() = 0;

    // Cleans up any resources used by the benchmark
    virtual void CleanUp() {}

  protected:
    IMicrobenchmark() = default;
};
