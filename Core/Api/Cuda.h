#pragma once
#include "Defs.h"

#include <cuda_runtime.h>


// Wrap CUDA functions and types as stand alone structure for easier integration with microbenchmarks
struct Cuda {
  Cuda() = delete;
  Cuda(const Cuda &) = delete;
  Cuda &operator =(const Cuda &) = delete;

  using cudaError_t    = ::cudaError_t;
  using cudaEvent_t    = ::cudaEvent_t;
  using cudaDeviceProp = ::cudaDeviceProp;
  using cudaDeviceAttr = ::cudaDeviceAttr;
  using cudaMemcpyKind = ::cudaMemcpyKind;
  using cudaStream_t   = ::cudaStream_t;

  static constexpr cudaError_t cudaSuccess = ::cudaSuccess;

  static constexpr cudaDeviceAttr cudaDevAttrClockRate                         = ::cudaDevAttrClockRate;
  static constexpr cudaDeviceAttr cudaDevAttrMemoryClockRate                   = ::cudaDevAttrMemoryClockRate;
  static constexpr cudaDeviceAttr cudaDevAttrSingleToDoublePrecisionPerfRatio  = ::cudaDevAttrSingleToDoublePrecisionPerfRatio;
  static constexpr cudaDeviceAttr cudaDevAttrMaxSharedMemoryPerMultiProcessor  = ::cudaDevAttrMaxSharedMemoryPerMultiprocessor;

  //--------------
  //--- Errors ---
  //--------------

  static cudaError_t cudaGetLastError() {
    return ::cudaGetLastError();
  }

  static const char *cudaGetErrorString(cudaError_t error) {
    return ::cudaGetErrorString(error);
  }

  //---------------
  //--- Devices ---
  //---------------

  static cudaError_t cudaDriverGetVersion(int *version) {
    return ::cudaDriverGetVersion(version);
  }
  
  static cudaError_t cudaRuntimeGetVersion(int *version) {
    return ::cudaRuntimeGetVersion(version);
  }

  static cudaError_t cudaGetDeviceCount(int *count) {
    return ::cudaGetDeviceCount(count);
  }

  static cudaError_t cudaSetDevice(int device) {
    return ::cudaSetDevice(device);
  }

  static cudaError_t cudaGetDeviceProperties(cudaDeviceProp *prop, int device) {
    return ::cudaGetDeviceProperties(prop, device);
  }

  static cudaError_t cudaDeviceGetAttribute(int *value, cudaDeviceAttr attr, int device) {
    return ::cudaDeviceGetAttribute(value, attr, device);
  }

  static cudaError_t cudaDeviceSynchronize() {
    return ::cudaDeviceSynchronize();
  }

  //--------------
  //--- Memory ---
  //--------------

  template<class T>
  static cudaError_t cudaMalloc(T **devPtr, size_t size) {
    return ::cudaMalloc(devPtr, size);
  }
  
  template<class T>
  static cudaError_t cudaFree(T *devPtr) {
    return ::cudaFree(devPtr);
  }

  template<class T>
  static cudaError_t cudaMemcpy(T *dst, const T *src, size_t count, cudaMemcpyKind kind) {
    return ::cudaMemcpy(dst, src, count, kind);
  }

  static cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream = 0) {
    return ::cudaEventRecord(event, stream);
  }

  //--------------
  //--- Events ---
  //--------------

  static cudaError_t cudaEventCreate(cudaEvent_t *event) {
    return ::cudaEventCreate(event);
  }

  static cudaError_t cudaEventDestroy(cudaEvent_t event) {
    return ::cudaEventDestroy(event);
  }

  static cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end) {
    return ::cudaEventElapsedTime(ms, start, end);
  }

  static cudaError_t cudaEventSynchronize(cudaEvent_t event) {
    return ::cudaEventSynchronize(event);
  }
};
