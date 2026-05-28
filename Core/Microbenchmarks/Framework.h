#pragma once
#include "Defs.h"

#include "IMicrobenchmark.h"

// Detects available CUDA devices, manages and runs microbenchmarks
// May be configured for various scenarios
class Framework {
  public:
    Framework(size_t n_iterations, size_t n_warmup, size_t n_outliers);

    // Sets new stream for user-friendly text output
    // By default, this stream is disabled
    void SetTextStream(std::ostream &text) { text_stream_ = text; }

    // Sets new stream for CSV output
    // By default, this stream is disabled
    void SetCsvStream(std::ostream &csv) { csv_stream_ = csv; }

    // Sets new tag that will be assigned to all following benchmarks
    // Benchmarks with the same tag will be grouped together in the output
    void SetTag(const std::string &tag) { tag_ = tag; }

    // Measures some kind of latency: overheads, delay before receiving the result, etc
    // This version expects a benchmark that works with blocks of different sizes
    void AddLatencyBenchmark(std::unique_ptr<IMicrobenchmark<size_t>> &&test,
                             const std::vector<size_t> &block_sizes);

    // This version expects a benchmark that does not need any arguments
    void AddLatencyBenchmark(std::unique_ptr<IMicrobenchmark<>> &&test);

    // Runs all benchmarks that were added, immediately prints the results to streams
    void Run();

  private:
    size_t n_iterations_, n_warmup_, n_outliers_;
    std::reference_wrapper<std::ostream> text_stream_;
    std::reference_wrapper<std::ostream> csv_stream_;

    std::string tag_;
};
