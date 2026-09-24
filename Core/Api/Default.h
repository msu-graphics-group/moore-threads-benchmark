#pragma once
#include "Defs.h"

// Recognize the target API and set defines
// Neither CUDA nor MUSA nor HIP guarantees that FP16 is available
#if defined(API_CUDA)

  #include "Api/Cuda.h"
  #if defined(__has_include) && __has_include(<cuda_fp16.h>)
    #include <cuda_fp16.h>
    #define FP16_SUPPORT
  #endif

  using Api = Cuda;
  constexpr inline std::string_view ApiName() { return "CUDA"; }

#elif defined(API_HIP)

  #include "Api/Hip.h"
  #if defined(__has_include) && __has_include(<hip_fp16.h>)
    #include <hip_fp16.h>
    #define FP16_SUPPORT
  #endif

  using Api = Hip;
  constexpr inline std::string_view ApiName() { return "HiP"; }

#elif defined(API_MUSA)

  #include "Api/Musa.h"
  #if defined(__has_include) && __has_include(<musa_fp16.h>)
    #include <musa_fp16.h>
    #define FP16_SUPPORT
  #endif

  using Api = Musa;
  constexpr inline std::string_view ApiName() { return "MUSA"; }

#else
  #error API not specified, you should recompile benchmark with the option like '-DAPI=CUDA' or '-DAPI=MUSA'
#endif


// Some helpers to simplify the code
constexpr inline bool IsCuda() { return ApiName() == "CUDA"; }
constexpr inline bool IsHip()  { return ApiName() == "HiP"; }
constexpr inline bool IsMusa() { return ApiName() == "MUSA"; }

constexpr inline bool IsFp16Supported() {
#if defined(FP16_SUPPORT)
  return true;
#else
  return false;
#endif
}

static inline void HandleError(Api::cudaError_t err, const char* file, int line)
{
  if (err != Api::cudaSuccess)
  {
    std::ostringstream oss;
    oss << file << ":" << line << ": " << Api::cudaGetErrorString(err);
    throw std::runtime_error(oss.str());
  }
}
#define HANDLE_ERROR(err) do { HandleError((err), __FILE__, __LINE__); } while (false)
