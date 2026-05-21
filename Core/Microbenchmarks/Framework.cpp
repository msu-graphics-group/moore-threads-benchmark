#include "Framework.h"

namespace {

// Vibe-coded by Gemini, implements '/dev/null' for C++ streams
class NullBuffer : public std::streambuf {
  public:
    int overflow(int c) override {
        return traits_type::not_eof(c);
    }
};
static NullBuffer null_buffer;
static std::ostream null(&null_buffer);

// Represents performance results after their averaging
// Units depend on the related benchmark's specifics
struct Results {
  double min{};
  double max{};
  double mean{};
  double stdev{};
  size_t n{};
};

// Removes warmup iterations outliers, performs statistical averaging of benchmark results
Results Average(std::vector<double> values, size_t n_warmup, size_t n_outliers) {
  assert(values.size() > n_warmup + n_outliers * 2);
  values = std::vector<double>(values.begin() + n_warmup, values.end());
  std::sort(values.begin(), values.end());
  values = std::vector<double>(values.begin() + n_outliers, values.end() - n_outliers);
  assert(!values.empty());

  Results results{};
  results.n = values.size();
  results.min = values.front();
  results.max = values.back();

  double sum = 0.0;
  for (double value : values) {
    sum += value;
  }
  results.mean = sum / results.n;

  double sum_sq_diff = 0.0;
  for (double value : values) {
    double diff = value - results.mean;
    sum_sq_diff += diff * diff;
  }
  results.stdev = std::sqrt(sum_sq_diff / results.n);

  return results;
}

} // unnamed namespace


Framework::Framework(size_t n_iterations, size_t n_warmup, size_t n_outliers)
  : n_iterations_(n_iterations), n_warmup_(n_warmup), n_outliers_(n_outliers),
    text_stream_(null), csv_stream_(null) {
}
