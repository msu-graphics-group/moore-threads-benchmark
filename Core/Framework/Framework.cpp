#include "Framework.h"

#include "Api/Default.h"
#include "DeviceInfo.h"

namespace {

// Implements '/dev/null' for C++ streams
// Coded by Gemini
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
Results Average(std::vector<double> values, double warmup_fraction, double outlier_fraction) {
  assert(!values.empty());
  assert(warmup_fraction >= 0.0);
  assert(outlier_fraction >= 0.0);

  // Convert fractions to the exact number of iterations
  size_t n_warmup  = (size_t)std::round(values.size() * warmup_fraction);
  size_t n_outliers = (size_t)std::round(values.size() * outlier_fraction);
  n_outliers = std::min(n_outliers, values.size() - 1);
  if (n_warmup + n_outliers >= values.size()) {
    n_warmup = values.size() - n_outliers - 1;
  }
  size_t n_min_outliers = n_outliers / 2;
  size_t n_max_outliers = n_outliers - n_min_outliers;
  assert(values.size() > n_warmup + n_outliers);

  // Average the results
  values = std::vector<double>(values.begin() + n_warmup, values.end());
  std::sort(values.begin(), values.end());
  values = std::vector<double>(values.begin() + n_min_outliers, values.end() - n_max_outliers);
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

std::string FormatResults(const ISealedBenchmark &bench, const Results &min, const Results &max) {
  std::ostringstream oss;
  switch (bench.Better()) {
    case WhoIsBetter::LowerIsBetter:
      oss << ToString(std::make_pair(min.mean, bench.Units()));
      break;

    case WhoIsBetter::HigherIsBetter:
      oss << ToString(std::make_pair(max.mean, bench.Units()));
      break;

    case WhoIsBetter::NeedMinMax:
      {
        std::string lo = ToString(std::make_pair(min.mean, bench.Units()));
        std::string hi = ToString(std::make_pair(max.mean, bench.Units()));
        if (lo == hi) {
          oss << lo;
        }
        else {
          oss << lo << " - " << hi;
        }
      }
      break;

    default:
      throw std::runtime_error("Unknown benchmark metrics");
  }

  return oss.str();
}

void PrintBenchmarkSection(const std::string &name, std::ostream &text_stream, std::ostream &csv_stream) {
  text_stream << std::endl;
  text_stream << "### " << name << " ###" << std::endl;
  text_stream << std::endl;
}

void PrintBenchmarkResults(const std::string &name, const std::string &results,
                           std::ostream &text_stream, std::ostream &csv_stream) {
  text_stream << "   " << name << ": " << results << "   " << std::endl;
}

} // unnamed namespace


Framework::Framework()
  : text_stream_(null), csv_stream_(null) {
}

void Framework::ExcludeIterations(double warm_up_fraction, double outlier_fraction) {
  if (warm_up_fraction >= 1.0 || warm_up_fraction < 0.0 ||
      outlier_fraction >= 1.0 || outlier_fraction < 0.0 ||
      warm_up_fraction + outlier_fraction >= 1.0) {
     throw std::invalid_argument("Invalid fraction of warm-up and/or outliers");
  }
  warm_up_fraction_ = warm_up_fraction;
  outlier_fraction_ = outlier_fraction;
}

void Framework::Run() {
  PrintDeviceInfo(0, text_stream_, csv_stream_);

  auto &out = text_stream_.get();
  for (auto &rec : benchmarks_) {
    PrintBenchmarkSection(rec.first, text_stream_, csv_stream_);

    for (auto &bench : rec.second) {
      std::optional<Results> min, max;

      bench->Reset();
      while (bench->MoveNext()) {
        try {
          auto results = Average(bench->Run(), warm_up_fraction_, outlier_fraction_);
          if (!min || results.mean < min->mean) {
            min = results;
          }
          if (!max || results.mean > max->mean) {
            max = results;
          }
        }
        catch (std::exception &) {
          // TODO: implement logging
          HANDLE_ERROR(Api::cudaDeviceReset());
        }
      }

      std::string results = min && max ? FormatResults(*bench, *min, *max) : "error";
      PrintBenchmarkResults(bench->Name(), results, text_stream_, csv_stream_);
    } // for benchmark
  } // for record
}
