#include "DeviceInfo.h"

#include "Api/Default.h"

#include <cassert>
#include <cstring>
#include <iomanip>
#include <sstream>

namespace {

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

// Returns 'yes' and 'no' instead of '1' and '0'
std::string YesOrNo(int value) {
  assert(value == 0 || value == 1);
  return value == 0 ? "no" : "yes";
}

} // unnamed namespace


DeviceInfo GetDeviceInfo(int device) {
  int device_count{};
  HANDLE_ERROR(Api::cudaGetDeviceCount(&device_count));
  assert(device <= device_count);

  // Get information about driver and runtime
  int driverVersion{}, runtimeVersion{};
  HANDLE_ERROR(Api::cudaDriverGetVersion(&driverVersion));
  HANDLE_ERROR(Api::cudaRuntimeGetVersion(&runtimeVersion));

  // Get static technical specifications
  Api::cudaDeviceProp props;
  std::memset(&props, 0, sizeof(props));
  HANDLE_ERROR(Api::cudaGetDeviceProperties(&props, device));

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

  // Wrap all values as 'DeviceInfo'
  std::vector<DeviceInfo::Parameter> params;
  DeviceInfo res { .name = props.name };

  // API
  params = {};
  params.emplace_back(DeviceInfo::Parameter("Driver version", std::to_string(driverVersion)));
  params.emplace_back(DeviceInfo::Parameter("Runtime version", std::to_string(runtimeVersion)));
  res.specifications.emplace_back(DeviceInfo::ParameterCategory("API", std::move(params)));

  // Device
  params = {};
  params.emplace_back(DeviceInfo::Parameter("Compute Capability",
                                      std::to_string(props.major) + '.' + std::to_string(props.minor)));
#if defined(API_HIP)
  params.emplace_back(DeviceInfo::Parameter("gcnArchName", props.gcnArchName));
#endif
  params.emplace_back(DeviceInfo::Parameter("multiProcessorCount", std::to_string(props.multiProcessorCount)));
  params.emplace_back(DeviceInfo::Parameter("clockRate", ApplyUnits(clockRate, 1e-3, "MHz")));
#if defined(API_HIP)
  params.emplace_back(DeviceInfo::Parameter("clockInstructionRate", ApplyUnits(props.clockInstructionRate, 1e-3, "MHz")));
#endif
  params.emplace_back(DeviceInfo::Parameter("integrated", YesOrNo(props.integrated)));
  params.emplace_back(DeviceInfo::Parameter("concurrentKernels", YesOrNo(props.concurrentKernels)));
  res.specifications.emplace_back(DeviceInfo::ParameterCategory("Device", std::move(params)));

  // Multiprocessor
  params = {};
  params.emplace_back(DeviceInfo::Parameter("warpSize", std::to_string(props.warpSize)));
#if !defined(API_HIP)
  params.emplace_back(DeviceInfo::Parameter("singleToDoublePrecisionPerfRatio", std::to_string(singleToDoublePrecisionPerfRatio)));
#endif
  params.emplace_back(DeviceInfo::Parameter("regsPerMultiprocessor", std::to_string(props.regsPerMultiprocessor)));
  params.emplace_back(DeviceInfo::Parameter("maxBlocksPerMultiProcessor", std::to_string(props.maxBlocksPerMultiProcessor)));
  params.emplace_back(DeviceInfo::Parameter("sharedMemPerMultiprocessor", ApplyUnits(props.sharedMemPerMultiprocessor)));
  params.emplace_back(DeviceInfo::Parameter("maxThreadsPerMultiProcessor", std::to_string(props.maxThreadsPerMultiProcessor)));
  params.emplace_back(DeviceInfo::Parameter("maxSharedMemoryPerMultiProcessor", ApplyUnits(maxSharedMemoryPerMultiProcessor)));
  res.specifications.emplace_back(DeviceInfo::ParameterCategory("Multiprocessor", std::move(params)));

  // Global memory
  params = {};
  params.emplace_back(DeviceInfo::Parameter("totalGlobalMem", ApplyUnits(props.totalGlobalMem)));
  params.emplace_back(DeviceInfo::Parameter("memoryClockRate", ApplyUnits(memoryClockRate, 1e-3, "MHz")));
  params.emplace_back(DeviceInfo::Parameter("memoryBusWidth", std::to_string(props.memoryBusWidth)));
  params.emplace_back(DeviceInfo::Parameter("ECCEnabled", YesOrNo(props.ECCEnabled)));
  params.emplace_back(DeviceInfo::Parameter("managedMemory", YesOrNo(props.managedMemory)));
  params.emplace_back(DeviceInfo::Parameter("unifiedAddressing", YesOrNo(props.unifiedAddressing)));
  params.emplace_back(DeviceInfo::Parameter("pageableMemoryAccess", YesOrNo(props.pageableMemoryAccess)));
  res.specifications.emplace_back(DeviceInfo::ParameterCategory("Global memory", std::move(params)));

  // Grid
  params = {};
  params.emplace_back(DeviceInfo::Parameter("maxGridSize",
      std::to_string(props.maxGridSize[0]) + " x " +
      std::to_string(props.maxGridSize[1]) + " x " +
      std::to_string(props.maxGridSize[2])));
  params.emplace_back(DeviceInfo::Parameter("sharedMemPerBlock", ApplyUnits(props.sharedMemPerBlock)));
  params.emplace_back(DeviceInfo::Parameter("regsPerBlock", std::to_string(props.regsPerBlock)));
  params.emplace_back(DeviceInfo::Parameter("maxThreadsPerBlock", std::to_string(props.maxThreadsPerBlock)));
  res.specifications.emplace_back(DeviceInfo::ParameterCategory("Grid", std::move(params)));

  // Caches
  params = {};
  params.emplace_back(DeviceInfo::Parameter("totalConstMem", ApplyUnits(props.totalConstMem)));
  params.emplace_back(DeviceInfo::Parameter("l2CacheSize", ApplyUnits(props.l2CacheSize)));
  params.emplace_back(DeviceInfo::Parameter("persistingL2CacheMaxSize",
      ApplyUnits(props.persistingL2CacheMaxSize, 100.0 / props.l2CacheSize, "%")));
  params.emplace_back(DeviceInfo::Parameter("localL1CacheSupported", YesOrNo(props.localL1CacheSupported)));
  params.emplace_back(DeviceInfo::Parameter("globalL1CacheSupported", YesOrNo(props.globalL1CacheSupported)));
  res.specifications.emplace_back(DeviceInfo::ParameterCategory("Caches", std::move(params)));

  return res;
}
