#pragma once
#include "Defs.h"

// Estimated peak performance of the selected device
// Note: these estimates may be inaccurate due to various factors:
//   - GPU clock boost states
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
  // Represents a device parameter with its technical name and user-friendly display value
  // Example: { "totalGlobalMem", "16.0 GB" }
  using Parameter = std::pair<std::string, std::string>;

  // Groups related device parameters into a categorized structure
  // Example: { "Global memory", { {"totalGlobalMem", "16.0 GB"}, {"memoryClockRate", "1200 MHz"} } }
  using ParameterCategory = std::pair<std::string, std::vector<Parameter>>;

  // Name of the device as returned by the underlying API
  std::string name;

  TheoreticalPerformance performance;

  // Technical specifications of the device organized by category (e.g., 'Multiprocessor', 'Caches', etc)
  // Since CUDA, HiP, and MUSA APIs have some differences, some parameters may be missing
  std::vector<ParameterCategory> specifications;
};

// Retrieves detailed information about a specific GPU device
DeviceInfo GetDeviceInfo(int device);
