#include "Units.h"


std::string ToString(double value, Unit unit) {
  return ToString(std::make_pair(value, unit));
}

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

    case Unit::Hz:
      prefixes = { {1e9, "GHz"}, {1e6, "MHz"}, {1e3, "KHz"}, {1.0, "Hz"} };
      break;

    case Unit::Bytes:
      prefixes = { {1024 * 1024 * 1024 * 1024.0, "TB"}, {1024 * 1024 * 1024.0, "GB"},
                   {1024 * 1024.0, "MB"}, {1024.0, "KB"}, {1.0, "Bytes"} };
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
  double scaled_value = v / chosen->first;
  bool need_dot = (int)(std::round(scaled_value * 10) - std::round(scaled_value) * 10) != 0;
  oss << std::fixed << std::setprecision(need_dot ? 1 : 0) << scaled_value << ' ' << chosen->second;
  return oss.str();
}
