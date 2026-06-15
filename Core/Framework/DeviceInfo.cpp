#include "DeviceInfo.h"

#include "Units.h"

//-------------------------------
//--- EstimateCudaPerformance ---
//-------------------------------

namespace {

enum class NvidiaArchitecture {
  Unknown,
  Kepler,    // 2012 (GK)
  Maxwell,   // 2014 (GM)
  Pascal,    // 2016 (GP)
  Volta,     // 2017 (GV)
  Turing,    // 2018 (TU)
  Ampere,    // 2020 (GA)
  Ada,       // 2022 (AD)
  Hopper,    // 2022 (GH)
  Blackwell  // 2024 (GB)
};

// Coded by DeepSeek, reviewed by Gemini, Qwen and me
// These formulas have several flaws, need to verify them with real GPUs
std::optional<TheoreticalPerformance> EstimateCudaPerformance(const Api::cudaDeviceProp &props,
                                                              int core_clock_rate, int memory_clock_rate) {
    TheoreticalPerformance perf;

    NvidiaArchitecture arch{ NvidiaArchitecture::Unknown };
    int simt_cores_per_sm{};
    int tensor_cores_per_sm{};
    int fp32_per_simt_core{};
    int fp16_per_simt_core{};
    double fp64_per_simt_core{};
    int fp16_per_tensor_core{};
    int fp32_per_tensor_core{};
    int fp64_per_tensor_core{};
    int data_rate_multiplier{2};

    if (props.major == 3) {
      arch                  = NvidiaArchitecture::Kepler;
      simt_cores_per_sm     = 192;
      tensor_cores_per_sm   = 0;

      fp32_per_simt_core    = 2;
      fp16_per_simt_core    = 0;
      fp64_per_simt_core    = (props.minor == 5) ? (2.0 / 3.0) : (2.0 / 24.0);
      fp16_per_tensor_core  = 0;
      fp32_per_tensor_core  = 0;
      fp64_per_tensor_core  = 0;
    }
    else if (props.major == 5) {
      arch                  = NvidiaArchitecture::Maxwell;
      simt_cores_per_sm     = 128;
      tensor_cores_per_sm   = 0;

      fp32_per_simt_core    = 2;
      fp16_per_simt_core    = 0;
      fp64_per_simt_core    = 2.0 / 32.0;
      fp16_per_tensor_core  = 0;
      fp32_per_tensor_core  = 0;
      fp64_per_tensor_core  = 0;
    }
    else if (props.major == 6) {
      arch                  = NvidiaArchitecture::Pascal;
      tensor_cores_per_sm   = 0;

      if (props.minor == 0) {
        simt_cores_per_sm   = 64;
        fp32_per_simt_core  = 2;
        fp16_per_simt_core  = 4;
        fp64_per_simt_core  = 1.0;
        data_rate_multiplier = 1;
       } 
       else {
        simt_cores_per_sm   = 128;
        fp32_per_simt_core  = 2;
        fp16_per_simt_core  = 2 / 64.0;
        fp64_per_simt_core  = 2 / 32.0;
      }
    }
    else if (props.major == 7) {
      fp32_per_simt_core    = 2;
      fp16_per_simt_core    = 4;

      if (props.minor == 0 || props.minor == 2) {
        arch = NvidiaArchitecture::Volta;
        fp64_per_simt_core  = 1;
      }
      else {
        arch = NvidiaArchitecture::Turing;
        fp64_per_simt_core  = 2.0 / 32.0;
      }

      simt_cores_per_sm     = 64;
      tensor_cores_per_sm   = 8;
      fp16_per_tensor_core  = 128;
      fp32_per_tensor_core  = 0;
      fp64_per_tensor_core  = 0;
    }
    else if (props.major == 8) {
      if (props.minor == 9) {
        arch                = NvidiaArchitecture::Ada;
        simt_cores_per_sm   = 128;
        tensor_cores_per_sm = 4;

        fp32_per_simt_core  = 2;
        fp16_per_simt_core  = 2;
        fp64_per_simt_core  = 2.0 / 64.0;
        fp16_per_tensor_core = 512;
        fp32_per_tensor_core = 256;
        fp64_per_tensor_core = 0;
        //data_rate_multiplier = 4;  // Need to differ GDDR6 from GDDR6X by GPU name
      }
      else {
        arch                = NvidiaArchitecture::Ampere;
        if (props.minor == 0) {
          simt_cores_per_sm = 64;
        } else {
          simt_cores_per_sm = 128;
        }
        tensor_cores_per_sm = 4;

        fp32_per_simt_core  = 2;
        fp16_per_simt_core  = 4;
        fp64_per_simt_core  = (props.minor == 0) ? (2.0 * 0.5) : (2.0 / 64.0);
        fp16_per_tensor_core = 512;
        fp32_per_tensor_core = 512;
        fp64_per_tensor_core = (props.minor == 0) ? 16 : 0;
        data_rate_multiplier = (props.minor == 0) ? 1 : 4;
      }
    }
    else if (props.major == 9) {
      arch                  = NvidiaArchitecture::Hopper;
      simt_cores_per_sm     = 128;
      tensor_cores_per_sm   = 4;

      fp32_per_simt_core    = 2;
      fp16_per_simt_core    = 4;
      fp64_per_simt_core    = 1;
      fp16_per_tensor_core  = 512;
      fp32_per_tensor_core  = 512;
      fp64_per_tensor_core  = 32;
      data_rate_multiplier  = 1;
    }
    else if (props.major == 10) {
      arch                  = NvidiaArchitecture::Blackwell;
      simt_cores_per_sm     = 128;
      tensor_cores_per_sm   = 4;

      fp32_per_simt_core    = 2;
      fp16_per_simt_core    = 4;
      fp64_per_simt_core    = 2.0 / 128.0;
      fp16_per_tensor_core  = 1024;
      fp32_per_tensor_core  = 1024;
      fp64_per_tensor_core  = 0;
      data_rate_multiplier  = (props.minor == 0) ? 1 : 3;
    }
    else {
      arch                  = NvidiaArchitecture::Unknown;
      simt_cores_per_sm     = 0;
      tensor_cores_per_sm   = 0;

      fp32_per_simt_core    = 0;
      fp16_per_simt_core    = 0;
      fp64_per_simt_core    = 0.0;
      fp16_per_tensor_core  = 0;
      fp32_per_tensor_core  = 0;
      fp64_per_tensor_core  = 0;
    }

    perf.simt_cores = props.multiProcessorCount * simt_cores_per_sm;
    perf.simt_fp32 = (uint64_t)perf.simt_cores * fp32_per_simt_core * core_clock_rate * 1000;
    perf.simt_fp16 = (uint64_t)perf.simt_cores * fp16_per_simt_core * core_clock_rate * 1000;
    perf.simt_fp64 = (uint64_t)(perf.simt_cores * fp64_per_simt_core * core_clock_rate * 1000);

    perf.tensor_cores = props.multiProcessorCount * tensor_cores_per_sm;
    perf.tensor_fp16 = (uint64_t)perf.tensor_cores * fp16_per_tensor_core * core_clock_rate * 1000;
    perf.tensor_fp32 = (uint64_t)perf.tensor_cores * fp32_per_tensor_core * core_clock_rate * 1000;
    perf.tensor_fp64 = (uint64_t)perf.tensor_cores * fp64_per_tensor_core * core_clock_rate * 1000;

    perf.memory_bandwidth = (uint64_t)data_rate_multiplier * props.memoryBusWidth / 8 * memory_clock_rate * 1000;
    
    return arch == NvidiaArchitecture::Unknown ? std::optional<TheoreticalPerformance>() : perf;
}

} // unnamed namespace


//-------------------------------
//--- EstimateMusaPerformance ---
//-------------------------------

enum class MooreThreadsArchitecture {
  Unknown,
  Sudi,
  Chunxiao
};

std::optional<TheoreticalPerformance> EstimateMusaPerformance(const Api::cudaDeviceProp &props,
                                                              int core_clock_rate, int memory_clock_rate) {
  MooreThreadsArchitecture arch{ MooreThreadsArchitecture::Unknown };
  int tensor_cores_per_mp{};
  double fp64_per_simt_core{};
  if (props.major == 1 && props.minor == 0) {
    arch = MooreThreadsArchitecture::Sudi;
  }
  else if (props.major == 2 && props.minor == 1) {
    arch = MooreThreadsArchitecture::Chunxiao;
    tensor_cores_per_mp = 4;
    fp64_per_simt_core = 1.0 / 64;
  }

  TheoreticalPerformance perf;
  perf.simt_cores   = props.multiProcessorCount * 128;
  perf.tensor_cores = props.multiProcessorCount * tensor_cores_per_mp;
  perf.simt_fp16    = 0;
  perf.simt_fp32    = perf.simt_cores * 2 * core_clock_rate * 1000;
  perf.simt_fp64    = perf.simt_cores * fp64_per_simt_core * core_clock_rate * 1000;
  perf.tensor_fp16  = perf.tensor_cores * 256 * core_clock_rate * 1000;
  perf.tensor_fp32  = 0;
  perf.tensor_fp64  = 0;
  perf.memory_bandwidth = (uint64_t)2 * props.memoryBusWidth / 8 * memory_clock_rate * 1000;

  return arch == MooreThreadsArchitecture::Unknown ? std::optional<TheoreticalPerformance>() : perf;
}


//---------------------
//--- GetDeviceInfo ---
//---------------------

namespace {

// Convertes some ration to a percentage string
std::string ToPercentage(double fraction) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(1) << fraction * 100 << '%';
  return oss.str();
}

// Returns 'yes' and 'no' instead of '1' and '0'
std::string YesOrNo(int value) {
  assert(value == 0 || value == 1);
  return value == 0 ? "no" : "yes";
}

} // unnamed namespace


DeviceInfo GetDeviceInfo(int device) {
  DeviceInfo res;

  int device_count{};
  HANDLE_ERROR(Api::cudaGetDeviceCount(&device_count));
  assert(device < device_count);

  // Get information about driver and runtime
  int driverVersion{}, runtimeVersion{};
  HANDLE_ERROR(Api::cudaDriverGetVersion(&driverVersion));
  HANDLE_ERROR(Api::cudaRuntimeGetVersion(&runtimeVersion));

  // Get static technical properties
  std::memset(&res.properties, 0, sizeof(res.properties));
  HANDLE_ERROR(Api::cudaGetDeviceProperties(&res.properties, device));
  const auto &props = res.properties;
  res.name = props.name;

  // Retrieve some dynamic attributes
  int clockRate{}, memoryClockRate{};
  int maxSharedMemoryPerMultiProcessor{};
  HANDLE_ERROR(Api::cudaDeviceGetAttribute(&clockRate, Api::cudaDevAttrClockRate, device));
  HANDLE_ERROR(Api::cudaDeviceGetAttribute(&memoryClockRate, Api::cudaDevAttrMemoryClockRate, device));
  HANDLE_ERROR(Api::cudaDeviceGetAttribute(&maxSharedMemoryPerMultiProcessor, Api::cudaDevAttrMaxSharedMemoryPerMultiProcessor, device));
#if !defined(API_HIP)
  int singleToDoublePrecisionPerfRatio{};
  HANDLE_ERROR(Api::cudaDeviceGetAttribute(&singleToDoublePrecisionPerfRatio, Api::cudaDevAttrSingleToDoublePrecisionPerfRatio, device));
#endif

  // Wrap all properties as parameters
  std::vector<DeviceInfo::Parameter> params;

  // API
  params = {};
  params.emplace_back(DeviceInfo::Parameter("Driver version",
                                            std::to_string(driverVersion)));
  params.emplace_back(DeviceInfo::Parameter("Runtime version",
                                            std::to_string(runtimeVersion)));
  res.parameters.emplace_back(DeviceInfo::ParameterCategory("API", std::move(params)));

  // Device
  params = {};
  params.emplace_back(DeviceInfo::Parameter("Compute Capability",
                                            std::to_string(props.major) + '.' + std::to_string(props.minor)));
#if defined(API_HIP)
  params.emplace_back(DeviceInfo::Parameter("gcnArchName",
                                            props.gcnArchName));
#endif
  params.emplace_back(DeviceInfo::Parameter("multiProcessorCount",
                                            std::to_string(props.multiProcessorCount)));
  params.emplace_back(DeviceInfo::Parameter("clockRate",
                                            ToString(clockRate * 1000, Unit::Hz)));
#if defined(API_HIP)
  params.emplace_back(DeviceInfo::Parameter("clockInstructionRate",
                                            ToString(props.clockInstructionRate * 1000, Unit::Hz)));
#endif
  params.emplace_back(DeviceInfo::Parameter("integrated",
                                            YesOrNo(props.integrated)));
  params.emplace_back(DeviceInfo::Parameter("concurrentKernels",
                                            YesOrNo(props.concurrentKernels)));
  res.parameters.emplace_back(DeviceInfo::ParameterCategory("Device", std::move(params)));

  // Multiprocessor
  params = {};
  params.emplace_back(DeviceInfo::Parameter("warpSize",
                                            std::to_string(props.warpSize)));
#if !defined(API_HIP)
  params.emplace_back(DeviceInfo::Parameter("singleToDoublePrecisionPerfRatio", std::to_string(singleToDoublePrecisionPerfRatio)));
#endif
  params.emplace_back(DeviceInfo::Parameter("regsPerMultiprocessor",
                                            std::to_string(props.regsPerMultiprocessor)));
  params.emplace_back(DeviceInfo::Parameter("maxBlocksPerMultiProcessor",
                                            std::to_string(props.maxBlocksPerMultiProcessor)));
  params.emplace_back(DeviceInfo::Parameter("sharedMemPerMultiprocessor",
                                            ToString(props.sharedMemPerMultiprocessor, Unit::Bytes)));
  params.emplace_back(DeviceInfo::Parameter("maxThreadsPerMultiProcessor",
                                            std::to_string(props.maxThreadsPerMultiProcessor)));
  params.emplace_back(DeviceInfo::Parameter("maxSharedMemoryPerMultiProcessor",
                                            ToString(maxSharedMemoryPerMultiProcessor, Unit::Bytes)));
  res.parameters.emplace_back(DeviceInfo::ParameterCategory("Multiprocessor", std::move(params)));

  // Global memory
  params = {};
  params.emplace_back(DeviceInfo::Parameter("totalGlobalMem",
                                            ToString(props.totalGlobalMem, Unit::Bytes)));
  params.emplace_back(DeviceInfo::Parameter("memoryClockRate",
                                            ToString(memoryClockRate * 1000.0, Unit::Hz)));
  params.emplace_back(DeviceInfo::Parameter("memoryBusWidth",
                                            std::to_string(props.memoryBusWidth)));
  params.emplace_back(DeviceInfo::Parameter("ECCEnabled", 
                                            YesOrNo(props.ECCEnabled)));
  params.emplace_back(DeviceInfo::Parameter("managedMemory",
                                            YesOrNo(props.managedMemory)));
  params.emplace_back(DeviceInfo::Parameter("unifiedAddressing",
                                            YesOrNo(props.unifiedAddressing)));
  params.emplace_back(DeviceInfo::Parameter("pageableMemoryAccess",
                                            YesOrNo(props.pageableMemoryAccess)));
  res.parameters.emplace_back(DeviceInfo::ParameterCategory("Global memory", std::move(params)));

  // Grid
  params = {};
  params.emplace_back(DeviceInfo::Parameter("maxGridSize",
                                            std::to_string(props.maxGridSize[0]) + " x " +
                                            std::to_string(props.maxGridSize[1]) + " x " +
                                            std::to_string(props.maxGridSize[2])));
  params.emplace_back(DeviceInfo::Parameter("sharedMemPerBlock",
                                            ToString(props.sharedMemPerBlock, Unit::Bytes)));
  params.emplace_back(DeviceInfo::Parameter("regsPerBlock",
                                            std::to_string(props.regsPerBlock)));
  params.emplace_back(DeviceInfo::Parameter("maxThreadsPerBlock",
                                            std::to_string(props.maxThreadsPerBlock)));
  res.parameters.emplace_back(DeviceInfo::ParameterCategory("Grid", std::move(params)));

  // Caches
  params = {};
  params.emplace_back(DeviceInfo::Parameter("totalConstMem",
                                            ToString(props.totalConstMem, Unit::Bytes)));
  params.emplace_back(DeviceInfo::Parameter("l2CacheSize",
                                            ToString(props.l2CacheSize, Unit::Bytes)));
  params.emplace_back(DeviceInfo::Parameter("persistingL2CacheMaxSize",
                                            ToPercentage((double)props.persistingL2CacheMaxSize / props.l2CacheSize)));
  params.emplace_back(DeviceInfo::Parameter("localL1CacheSupported",
                                            YesOrNo(props.localL1CacheSupported)));
  params.emplace_back(DeviceInfo::Parameter("globalL1CacheSupported",
                                            YesOrNo(props.globalL1CacheSupported)));
  res.parameters.emplace_back(DeviceInfo::ParameterCategory("Caches", std::move(params)));

  // Now, we need to estimate the theoretical performance
  if (IsCuda()) {
    res.performance = EstimateCudaPerformance(props, clockRate, memoryClockRate);
  }
  else if (IsMusa()) {
    res.performance = EstimateMusaPerformance(props, clockRate, memoryClockRate);
  }

  // Add estimates to parameters
  if (res.performance) {
    const auto &perf = res.performance.value();
    params = {};
    params.emplace_back(DeviceInfo::Parameter("SIMT (fp16)",   ToString(perf.simt_fp16, Unit::Flops)));
    params.emplace_back(DeviceInfo::Parameter("SIMT (fp32)",   ToString(perf.simt_fp32, Unit::Flops)));
    params.emplace_back(DeviceInfo::Parameter("SIMT (fp64)",   ToString(perf.simt_fp64, Unit::Flops)));
    params.emplace_back(DeviceInfo::Parameter("Tensor (fp16)", ToString(perf.tensor_fp16, Unit::Flops)));
    params.emplace_back(DeviceInfo::Parameter("Tensor (fp32)", ToString(perf.tensor_fp32, Unit::Flops)));
    params.emplace_back(DeviceInfo::Parameter("Tensor (fp64)", ToString(perf.tensor_fp64, Unit::Flops)));
    params.emplace_back(DeviceInfo::Parameter("Bandwidth",     ToString(perf.memory_bandwidth, Unit::BytesPerSecond)));
    res.parameters.emplace_back(DeviceInfo::ParameterCategory("Performance", std::move(params)));
  }

  return res;
}
