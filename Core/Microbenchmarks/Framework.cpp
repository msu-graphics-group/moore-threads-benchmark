#include "Framework.h"

#include "Api/Default.h"
#include "DeviceInfo.h"

#include <string>

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
  results.stdev = std::sqrt(sum_sq_diff / (std::max(results.n, (size_t)2) - 1));

  return results;
}

void PrintDeviceInfo(int device, std::ostream &text_stream, std::ostream &csv_stream) {
  auto info = GetDeviceInfo(device);

  text_stream << info.name << std::endl;

  size_t max_key_length = 0;
  for (const auto &category : info.parameters) {
    for (const auto &parameter : category.second) {
      max_key_length = std::max(max_key_length, parameter.first.size());
    }
  }

  for (const auto &category : info.parameters) {
    text_stream << "   " << category.first << std::endl;
    for (const auto &parameter : category.second) {
      std::string padded = parameter.first + ":" + std::string(max_key_length - parameter.first.size(), ' ');
      text_stream << "      " << padded << " " << parameter.second << std::endl;
    }
    text_stream << std::endl;
  }
}

} // unnamed namespace


Framework::Framework(size_t n_iterations, size_t n_warmup, size_t n_outliers)
  : n_iterations_(n_iterations), n_warmup_(n_warmup), n_outliers_(n_outliers),
    text_stream_(null), csv_stream_(null) {
}

void Framework::Run() {
  PrintDeviceInfo(0, text_stream_, csv_stream_);
}
