#pragma once
#include "Defs.h"

#if defined(API_CUDA)
  #include "Api/Cuda.h"
  using Api = Cuda;
#endif

static inline void HandleError(Api::cudaError_t err, const char* file, int line)
{
  if (err != Api::cudaError_t::cudaSuccess)
  {
    std::ostringstream oss;
    oss << file << ":" << line << ": " << Api::cudaGetErrorString(err);
    throw std::runtime_error(oss.str());
  }
}
#define HANDLE_ERROR(err) do { HandleError((err), __FILE__, __LINE__); } while (false)
