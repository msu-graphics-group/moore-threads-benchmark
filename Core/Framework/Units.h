#pragma once
#include "Defs.h"

// Standardized units for microbenchmark results and device performance metrics
enum class Unit {
  Seconds,           // Time measurement in seconds
  Flops,             // Floating point operations per second
  IntOps,            // Integer operations per second
  BytesPerSecond,    // Data transfer rate in bytes per second
  Cycles,            // SM cycles measured by 'clock64()'
  Hz,                // Frequency in hertz
  Bytes              // Data size in bytes
};

// Converts a value to a user-friendly string format
// E.g., { 100500, BytesPerSecond } -> '100.5 KB/s'
std::string ToString(double value, Unit unit);

// Version that takes a pair of {value, unit} 
std::string ToString(const std::pair<double, Unit> &value);

