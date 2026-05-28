#include "Framework.h"

#include "Api/Default.h"

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

// Formats some value to a human-readable format with units (e.g., '42.0 MB')
std::string ApplyUnits(size_t value, double multiplier, const std::string &units) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(2) << value * multiplier << ' ' << units;
  return oss.str();
}

// Special version for memory sizes
std::string ApplyUnits(size_t bytes) {
  constexpr size_t GB = 1024 * 1024 * 1024;
  constexpr size_t MB = 1024 * 1024;
  constexpr size_t KB = 1024;

  if (bytes >= GB) {
    return ApplyUnits(bytes, 1.0 / GB, "GB");
  } else if (bytes >= MB) {
    return ApplyUnits(bytes, 1.0 / MB, "MB");
  } else if (bytes >= KB) {
    return ApplyUnits(bytes, 1.0 / KB, "KB");
  } else {
    return ApplyUnits(bytes, 1.0, "Bytes");
  }
}

void PrintDeviceInfo(int device, std::ostream &text_stream, std::ostream &csv_stream) {
  int device_count{};
  HANDLE_ERROR(Api::cudaGetDeviceCount(&device_count));
  assert(device <= device_count);

  Api::cudaDeviceProp props;
  std::memset(&props, 0, sizeof(props));
  HANDLE_ERROR(Api::cudaGetDeviceProperties(&props, device));

  text_stream << props.name << std::endl;
  text_stream << "   Device:" << std::endl;
  text_stream << "      Compute Capability:                " << props.major << '.' << props.minor << std::endl;
  text_stream << "      multiProcessorCount:               " << props.multiProcessorCount << std::endl;
  text_stream << "      concurrentKernels:                 " << props.concurrentKernels << std::endl;
#if defined(API_HIP)
  text_stream << "      gcnArchName:                       " << props.gcnArchName << std::endl;
  text_stream << "      clockRate:                         " << ApplyUnits(props.clockRate, 1e-3, "MHz") << std::endl;
#endif
  text_stream << std::endl;
  text_stream << "   Multiprocessor:" << std::endl;
  text_stream << "      warpSize:                          " << props.warpSize << std::endl;
  text_stream << "      regsPerMultiprocessor:             " << props.regsPerMultiprocessor << std::endl;
  text_stream << "      maxBlocksPerMultiProcessor:        " << props.maxBlocksPerMultiProcessor << std::endl;
  text_stream << "      sharedMemPerMultiprocessor:        " << ApplyUnits(props.sharedMemPerMultiprocessor) << std::endl;
  text_stream << "      maxThreadsPerMultiProcessor:       " << props.maxThreadsPerMultiProcessor << std::endl;
#if (API_HIP)
  text_stream << "      maxSharedMemoryPerMultiProcessor:  " << ApplyUnits(props.maxSharedMemoryPerMultiProcessor) << std::endl;
#endif
  text_stream << std::endl;

  text_stream << "   Global memory" << std::endl;
  text_stream << "      totalGlobalMem:                    " << ApplyUnits(props.totalGlobalMem) << std::endl;
  text_stream << "      memoryBusWidth:                    " << props.memoryBusWidth << std::endl;
  text_stream << "      ECCEnabled:                        " << props.ECCEnabled << std::endl;
  text_stream << "      managedMemory:                     " << props.managedMemory << std::endl;
  text_stream << "      unifiedAddressing:                 " << props.unifiedAddressing << std::endl;
  text_stream << "      pageableMemoryAccess:              " << props.pageableMemoryAccess << std::endl;
#if defined(API_HIP)
  text_stream << "      memoryClockRate:                   " << ApplyUnits(props.memoryClockRate, 1e-3, "MHz") << std::endl;
#endif
  text_stream << std::endl;

  text_stream << "   Grid" << std::endl;
  text_stream << "      maxGridSize:                       " << props.maxGridSize[0] << " x "
              << props.maxGridSize[1] << " x " << props.maxGridSize[2] << std::endl;
  text_stream << "      sharedMemPerBlock:                 " << ApplyUnits(props.sharedMemPerBlock) << std::endl;
  text_stream << "      regsPerBlock:                      " << props.regsPerBlock << std::endl;
  text_stream << "      maxThreadsPerBlock:                " << props.maxThreadsPerBlock << std::endl;
  text_stream << std::endl;

  text_stream << "   Caches" << std::endl;
  text_stream << "      totalConstMem:                     " << ApplyUnits(props.totalConstMem) << std::endl;
  text_stream << "      l2CacheSize:                       " << ApplyUnits(props.l2CacheSize) << std::endl;
  text_stream << "      persistingL2CacheMaxSize:          "
              << ApplyUnits(props.persistingL2CacheMaxSize, 100.0 / props.l2CacheSize, "%") << std::endl;
  text_stream << "      localL1CacheSupported:             " << props.localL1CacheSupported << std::endl;
  text_stream << "      globalL1CacheSupported:            " << props.globalL1CacheSupported << std::endl;

  csv_stream << props.name  << std::endl;
}

} // unnamed namespace


Framework::Framework(size_t n_iterations, size_t n_warmup, size_t n_outliers)
  : n_iterations_(n_iterations), n_warmup_(n_warmup), n_outliers_(n_outliers),
    text_stream_(null), csv_stream_(null) {
}

void Framework::Run() {
  PrintDeviceInfo(0, text_stream_, csv_stream_);
}
