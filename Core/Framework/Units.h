#pragma once
#include "Defs.h"

// Units for results returned by microbenchmarks
enum class Unit {
  Seconds,           // Time measurement in seconds
  Flops,             // Floating point operations per second
  IntOps,            // Integer operations per second
  BytesPerSecond,    // Data transfer rate in bytes per second
  Cycles             // SM cycles measured by 'clock64()'
};

// Converts a value to a user-friendly string format
// E.g., { 100500, BytesPerSecond } -> '100.5 KB/s'
std::string ToString(const std::pair<double, Unit> &value);
