#pragma once
#include "Defs.h"

#if defined(API_CUDA)
  #include "Api/Cuda.h"
  using Api = Cuda;
  static inline std::string ApiName() { return "CUDA"; }
  constexpr bool IsCuda() { return true; }
  constexpr bool IsHip()  { return false; }
  constexpr bool IsMusa() { return false; }
#elif defined(API_HIP)
  #include "Api/Hip.h"
  using Api = Hip;
  static inline std::string ApiName() { return "HiP"; }
  constexpr bool IsCuda() { return false; }
  constexpr bool IsHip()  { return true; }
  constexpr bool IsMusa() { return false; }
#elif defined(API_MUSA)
  #include "Api/Musa.h"
  using Api = Musa;
  static inline std::string ApiName() { return "MUSA"; }
  constexpr bool IsCuda() { return false; }
  constexpr bool IsHip()  { return false; }
  constexpr bool IsMusa() { return true; }
#else
  #error API not specified, you should recompile benchmark with the option like '-DAPI=CUDA' or '-DAPI=MUSA'
#endif

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
