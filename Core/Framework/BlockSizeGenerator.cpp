#include "BlockSizeGenerator.h"


Api::cudaDeviceProp &BlockSizeGenerator::Props() {
  static int device = -1;
  static Api::cudaDeviceProp props;
  
  int cur_device{};
  HANDLE_ERROR(Api::cudaGetDevice(&cur_device));
  if (device != cur_device) {
    HANDLE_ERROR(Api::cudaGetDeviceProperties(&props, cur_device));
    device = cur_device;
  }

  return props;
}

size_t BlockSizeGenerator::MaxBlockSize(size_t bytes_per_element,
                                        double max_memory_usage,
                                        size_t max_allocated_memory) {
  assert(bytes_per_element >= 1);
  assert(max_memory_usage > 0.0);
  assert(max_memory_usage < 1.0);
  size_t limiter = static_cast<size_t>(max_memory_usage * Props().totalGlobalMem);
  return std::min(max_allocated_memory, limiter) / bytes_per_element;
}

// Coded by DeepSeek-v4
std::vector<size_t> BlockSizeGenerator::Uniform(size_t lo, size_t hi, size_t divisor, size_t n_blocks) {
  assert(lo <= hi);
  assert(divisor != 0);
  assert(n_blocks != 0);

  // Re-align 'lo' and 'hi' to the nearest multiple of 'divisor', check special cases
  lo = ((lo + divisor - 1) / divisor) * divisor;
  hi = (hi / divisor) * divisor;

  if (lo > hi) {
    throw std::runtime_error("no suitable blocks exist in the requested interval");
  }

  size_t total_count = (hi - lo) / divisor + 1;
  if (total_count < n_blocks) {
    throw std::runtime_error("the requested range does not contain enough blocks");
  }

  if (n_blocks == 1) {
    return { lo };
  }

  // Select 'n_blocks' equally‑spaced values from the re-aligned [lo, hi] range
  std::vector<size_t> res;
  res.reserve(n_blocks);

  double step = (double)(total_count - 1) / (double)(n_blocks - 1);
  for (size_t i = 0; i < n_blocks; i++) {
    size_t idx = (size_t)(std::round(i * step));
    idx = std::min(idx, total_count - 1);
    res.push_back(lo + idx * divisor);
  }

  return res;
}

// Coded by DeepSeek-v4
std::vector<size_t> BlockSizeGenerator::Exponential(size_t lo, size_t hi, size_t divisor, double base) {
  assert(lo <= hi);
  assert(divisor != 0);
  assert(base > 1.0);

  // Helper that snaps a value to the nearest multiple of 'divisor' while staying within [lo, hi]
  auto snap_nearest = [&](size_t val) -> size_t {
    const size_t lower = (val / divisor) * divisor;
    const size_t upper = lower + divisor;

    bool lower_ok = (lower >= lo && lower <= hi);
    bool upper_ok = (upper >= lo && upper <= hi);

    if (lower_ok && upper_ok) {
      return (val - lower <= upper - val) ? lower : upper;
    } else if (lower_ok) {
      return lower;
    } else if (upper_ok) {
      return upper;
    } else {
      return lo;
    }
  };

  // Spawn blocks with the requested sizes
  std::vector<size_t> res;
  double current = std::max(1.0, (double)lo);
  while (true) {
    if (current > hi) {
      break;
    }

    auto aligned = (size_t)(std::round(current / divisor) * divisor);
    res.push_back(snap_nearest(aligned));

    current *= base;
  }

  // Eliminate possible duplicates (they can arise from snapping) and sort
  std::sort(res.begin(), res.end());
  auto last = std::unique(res.begin(), res.end());
  res.erase(last, res.end());

  return res;
}

// Coded by DeepSeek-v4
std::vector<size_t> BlockSizeGenerator::Random(size_t lo, size_t hi, size_t divisor, size_t n_blocks, uint64_t seed) {
  assert(lo <= hi);
  assert(divisor != 0);

  // Re‑align lo and hi to multiples of divisor
  lo = ((lo + divisor - 1) / divisor) * divisor;
  hi = (hi / divisor) * divisor;

  if (lo > hi) {
    throw std::runtime_error("no suitable blocks exist in the requested interval");
  }

  size_t total_count = (hi - lo) / divisor + 1;
  if (n_blocks > total_count) {
    throw std::runtime_error("the requested range does not contain enough blocks");
  }

  std::mt19937_64 gen(seed);
  std::uniform_int_distribution<size_t> dist(0, total_count - 1);
  std::unordered_set<size_t> chosen;
  chosen.reserve(n_blocks);

  
  // Stage 1 – independent random sampling
  constexpr int max_attempts = 5;
  for (int attempt = 0; attempt < max_attempts && chosen.size() < n_blocks; attempt++) {
    size_t need = n_blocks - chosen.size();
    for (size_t j = 0; j < need; j++) {
      chosen.insert(lo + dist(gen) * divisor);
    }
  }

  // Stage 2 – if duplicates still prevent us from reaching n_blocks,
  // build the full list of remaining valid candidates and pick from it
  if (chosen.size() < n_blocks) {
    std::vector<size_t> remaining;
    remaining.reserve(total_count - chosen.size());

    for (size_t idx = 0; idx < total_count; ++idx) {
      size_t val = lo + idx * divisor;
      if (chosen.find(val) == chosen.end()) {
        remaining.push_back(val);
      }
    }

    // Shuffle the remaining values and take what we need
    std::shuffle(remaining.begin(), remaining.end(), gen);
    size_t need = n_blocks - chosen.size();
    for (size_t i = 0; i < need; ++i) {
      chosen.insert(remaining[i]);
    }
  }

  // Return a sorted vector for consistency
  std::vector<size_t> result(chosen.begin(), chosen.end());
  std::sort(result.begin(), result.end());
  return result;
}
