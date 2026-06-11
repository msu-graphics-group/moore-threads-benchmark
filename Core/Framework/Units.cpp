#include "Units.h"

#include <cmath>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

std::string ToString(const std::pair<double, Unit> &value) {
  double v = value.first;
  Unit u = value.second;

  std::vector<std::pair<double, std::string>> prefixes;
  switch (u) {
    case Unit::Seconds:
      prefixes = { {1.0,  "s"}, {1e-3, "ms"}, {1e-6, "us"}, {1e-9, "ns"} };
      break;

    case Unit::Flops:
      prefixes = { {1e12, "TFlops"}, {1e9,  "GFlops"}, {1e6,  "MFlops"}, {1e3,  "KFlops"}, {1.0,  "Flops"} };
      break;
      
    case Unit::IntOps:
      prefixes = { {1e12, "TIntOps"}, {1e9,  "GIntOps"}, {1e6,  "MIntOps"}, {1e3,  "KIntOps"}, {1.0,  "IntOps"}
      };
      break;

    case Unit::BytesPerSecond:
      prefixes = { {1e12, "TB/s"}, {1e9,  "GB/s"}, {1e6,  "MB/s"}, {1e3,  "KB/s"}, {1.0,  "B/s"} };
      break;

    case Unit::Cycles:
      prefixes = { {1e12, "TCycles"}, {1e9,  "GCycles"}, {1e6,  "MCycles"}, {1e3,  "KCycles"}, {1.0,  "Cycles"} };
      break;
  }

  std::optional<std::pair<double, std::string>> chosen;
  for (const auto &p : prefixes) {
    if (v >= p.first) {
      chosen = p;
      break;
    }
  }
  if (!chosen) {
    chosen = prefixes.back();
  }

  std::ostringstream oss;
  oss << std::fixed << std::setprecision(1)
      << std::round(v / chosen->first * 10.0) / 10.0 << ' ' << chosen->second;
  return oss.str();
}
