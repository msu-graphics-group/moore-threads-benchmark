#pragma once
#include "Defs.h"

#include "Api/Default.h"

// Estimated peak performance of the selected device
// Note: these estimates may be inaccurate due to various factors:
//   - GPU boosts clocks
//   - Unaccounted GPU architecture specifics
//   - Inaccuracies in our performance formulas
struct TheoreticalPerformance {
  // The total number of SIMT/tensor cores on the GPU
  size_t simt_cores{}, tensor_cores{};

  // Performance of regular SIMT cores
  // Measured in floating-point operations per second (flops)
  uint64_t simt_fp64{}, simt_fp32{}, simt_fp16{};

  // Performance of tensor cores
  // Measured in floating-point operations per second (flops)
  uint64_t tensor_fp64{}, tensor_fp32{}, tensor_fp16{};

  // Memory bandwidth of device global memory
  // Measured in bytes per second (B/s)
  uint64_t memory_bandwidth{};
};

struct DeviceInfo {
  // Represents a device parameter with its technical name and user-friendly value
  // Example: { "totalGlobalMem", "16.0 GB" }
  using Parameter = std::pair<std::string, std::string>;

  // Groups related device parameters into a categorized structure
  // Example: { "Global memory", { {"totalGlobalMem", "16.0 GB"}, {"memoryClockRate", "1200 MHz"} } }
  using ParameterCategory = std::pair<std::string, std::vector<Parameter>>;

  // Name of the device as returned by the underlying API
  std::string name;

  // The original and unmodified properties of the GPU as returned by the underlying API
  // Exactly the same copy of these properties can be acquired by calling 'Api::cudaGetDeviceProperties()'
  Api::cudaDeviceProp properties;

  // Estimates for peak performance of the GPU based on its architecture and clock rates
  // May be empty if we know nothing about this architecture
  std::optional<TheoreticalPerformance> performance;

  // Technical specifications of the device organized by category (e.g., 'Multiprocessor', 'Caches', etc)
  // They combine 'properties' and 'performance', enriching them with some dynamically changing attributes
  // Since CUDA, HiP, and MUSA APIs have some differences, some parameters may be missing or introduced
  std::vector<ParameterCategory> parameters;
};

// Retrieves detailed information about a specific GPU device
DeviceInfo GetDeviceInfo(int device);
