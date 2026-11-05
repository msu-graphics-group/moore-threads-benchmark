#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

// Host-side inputs, reference results and operation counts for the Tensor benchmark
// These calculations do not depend on a GPU runtime or device compiler
namespace tensor::model {

inline constexpr size_t kTile = 16;
inline constexpr size_t kTileElements = kTile * kTile;
inline constexpr size_t kWarpSize = 32;
inline constexpr size_t kWarpsPerBlock = 4;
inline constexpr size_t kBlockThreads = kWarpSize * kWarpsPerBlock;
inline constexpr size_t kMmaPerIteration = 32;
inline constexpr size_t kMaxIterations = 16384;
inline constexpr float kInputScale = 1.0f / 16.0f;
inline constexpr float kSeedScale = 1.0f / 256.0f;

inline constexpr bool IsValidIlp(size_t ilp) {
  return ilp == 1 || ilp == 2 || ilp == 4 || ilp == 8;
}

inline void CheckConfiguration(size_t ilp, size_t iterations) {
  if (!IsValidIlp(ilp)) {
    throw std::invalid_argument("Tensor ILP must be one of 1, 2, 4 or 8");
  }
  if (iterations == 0 || iterations > kMaxIterations) {
    throw std::invalid_argument("Tensor iterations must be between 1 and 16384");
  }
}

namespace detail {

// Unsigned arithmetic specifies the same bit pattern on every host. Separate salts
// keep A and B independent; the chain changes both matrices, not just the seed.
inline uint32_t Mix(uint32_t value) {
  value ^= value >> 16;
  value *= 0x7feb352du;
  value ^= value >> 15;
  value *= 0x846ca68bu;
  value ^= value >> 16;
  return value;
}

inline float Input(size_t chain, size_t row, size_t col, uint32_t salt) {
  const uint32_t key = salt ^ (static_cast<uint32_t>(chain) * 0x9e3779b9u)
                           ^ (static_cast<uint32_t>(row) * 0x85ebca6bu)
                           ^ (static_cast<uint32_t>(col) * 0xc2b2ae35u);
  return (Mix(key) & 1u) ? kInputScale : -kInputScale;
}

} // namespace detail

inline float InputA(size_t chain, size_t row, size_t col) {
  return detail::Input(chain, row, col, 0xa341316cu);
}

inline float InputB(size_t chain, size_t row, size_t col) {
  return detail::Input(chain, row, col, 0xc8013ea4u);
}

inline float InitialValue(size_t chain) {
  return static_cast<float>(chain + 1) * kSeedScale;
}

inline double Expected(size_t chain, size_t row, size_t col,
                       size_t ilp, size_t iterations) {
  CheckConfiguration(ilp, iterations);
  double dot = 0.0;
  for (size_t k = 0; k < kTile; ++k) {
    dot += static_cast<double>(InputA(chain, row, k)) * InputB(chain, k, col);
  }
  // All products and seeds are integer multiples of 1/256. The iteration limit
  // keeps even the worst-case numerator below 2^24, so FP32 additions stay exact.
  const double calls = static_cast<double>(iterations) * (kMmaPerIteration / ilp);
  return InitialValue(chain) + calls * dot;
}

inline double Operations(size_t warps, size_t iterations) {
  // A warp collectively performs each MMA. Neither its lanes nor ILP are extra
  // multipliers; the fixed 32 calls are divided among the independent chains.
  // Convert before multiplying, since a size_t product could overflow first.
  return static_cast<double>(warps) * static_cast<double>(iterations)
       * kMmaPerIteration * (2 * kTile * kTile * kTile);
}

inline size_t CalibrationIterations(double seconds, size_t probe_iterations = 32) {
  if (!std::isfinite(seconds) || seconds <= 0.0 || probe_iterations == 0) {
    throw std::invalid_argument("Tensor calibration needs a finite positive time and probe count");
  }
  constexpr size_t kMinIterations = 16;
  constexpr double kTargetSeconds = 0.02;
  const double scaled = (kTargetSeconds / seconds) * static_cast<double>(probe_iterations);
  // Clamp in floating point before conversion, including an overflow to infinity
  // for an extremely small (but positive) time measurement.
  if (scaled >= static_cast<double>(kMaxIterations)) {
    return kMaxIterations;
  }
  if (scaled <= static_cast<double>(kMinIterations)) {
    return kMinIterations;
  }
  return static_cast<size_t>(scaled);
}

} // namespace tensor::model
