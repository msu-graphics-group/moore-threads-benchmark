#pragma once

#include <musa_runtime.h>


struct Musa {
  Musa() = delete;
  Musa(const Musa &) = delete;
  Musa &operator=(const Musa &) = delete;

  using cudaError_t    = musaError_t;
  using cudaEvent_t    = musaEvent_t;
  using cudaDeviceProp = musaDeviceProp;
  using cudaDeviceAttr = musaDeviceAttr;
  using cudaMemcpyKind = musaMemcpyKind;
  using cudaStream_t   = musaStream_t;

  static constexpr cudaError_t cudaSuccess = musaSuccess;

  static constexpr cudaDeviceAttr cudaDevAttrClockRate                         = musaDevAttrClockRate;
  static constexpr cudaDeviceAttr cudaDevAttrMemoryClockRate                   = musaDevAttrMemoryClockRate;
  static constexpr cudaDeviceAttr cudaDevAttrSingleToDoublePrecisionPerfRatio  = musaDevAttrSingleToDoublePrecisionPerfRatio;
  static constexpr cudaDeviceAttr cudaDevAttrMaxSharedMemoryPerMultiProcessor  = musaDevAttrMaxSharedMemoryPerMultiprocessor;

  static constexpr unsigned int cudaHostAllocDefault                           = musaHostAllocDefault;

  static constexpr cudaMemcpyKind cudaMemcpyHostToDevice                       = ::musaMemcpyHostToDevice;
  static constexpr cudaMemcpyKind cudaMemcpyDeviceToHost                       = ::musaMemcpyDeviceToHost;
  static constexpr cudaMemcpyKind cudaMemcpyDeviceToDevice                     = ::musaMemcpyDeviceToDevice;

  //--------------
  //--- Errors ---
  //--------------

  static cudaError_t cudaGetLastError() {
    return ::musaGetLastError();
  }

  static const char *cudaGetErrorString(cudaError_t error) {
    return ::musaGetErrorString(error);
  }

  //---------------
  //--- Devices ---
  //---------------

  static cudaError_t cudaDriverGetVersion(int *version) {
    return ::musaDriverGetVersion(version);
  }
  
  static cudaError_t cudaRuntimeGetVersion(int *version) {
    return ::musaRuntimeGetVersion(version);
  }

  static cudaError_t cudaGetDeviceCount(int *count) {
    return ::musaGetDeviceCount(count);
  }

  static cudaError_t cudaSetDevice(int device) {
    return ::musaSetDevice(device);
  }

  static cudaError_t cudaGetDeviceProperties(cudaDeviceProp *prop, int device) {
    return ::musaGetDeviceProperties(prop, device);
  }

  static cudaError_t cudaDeviceGetAttribute(int *value, cudaDeviceAttr attr, int device) {
    return ::musaDeviceGetAttribute(value, attr, device);
  }

  static cudaError_t cudaDeviceSynchronize() {
    return ::musaDeviceSynchronize();
  }

  static cudaError_t cudaDeviceReset() {
    return ::musaDeviceReset();
  }

  //--------------
  //--- Memory ---
  //--------------

  template<class T>
  static cudaError_t cudaMalloc(T **devPtr, size_t size) {
    return ::musaMalloc(devPtr, size);
  }

  template<class T>
  static cudaError_t cudaMallocManaged(T **devPtr, size_t size) {
    return ::musaMallocManaged(devPtr, size);
  }

  template<class T>
  static cudaError_t cudaHostAlloc(T **devPtr, size_t size, unsigned int flags) {
    return ::musaHostAlloc(devPtr, size, unsigned int flags);
  }
  
  template<class T>
  static cudaError_t cudaFree(T *devPtr) {
    return ::musaFree(devPtr);
  }

  template<class T>
  static cudaError_t cudaFreeHost(T *devptr) {
    return ::musaFreeHost(devPtr);
  }

  template<class T>
  static cudaError_t cudaMemset(T *devPtr, int value, size_t count) {
    return ::musaMemset(devPtr, value, count);
  }

  template<class T>
  static cudaError_t cudaMemcpy(T *dst, const T *src, size_t count, cudaMemcpyKind kind) {
    return ::musaMemcpy(dst, src, count, kind);
  }

  template<class T>
  static cudaError_t cudaMemcpyAsync(T *dst, const T *src, size_t count,
                                     cudaMemcpyKind kind, cudaStream_t stream = 0) {
    return ::musaMemcpyAsync(dst, src, kind, stream);
  }

  //--------------
  //--- Events ---
  //--------------

  static cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream = 0) {
    return ::musaEventRecord(event, stream);
  }

  static cudaError_t cudaEventCreate(cudaEvent_t *event) {
    return ::musaEventCreate(event);
  }

  static cudaError_t cudaEventDestroy(cudaEvent_t event) {
    return ::musaEventDestroy(event);
  }

  static cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end) {
    return ::musaEventElapsedTime(ms, start, end);
  }

  static cudaError_t cudaEventSynchronize(cudaEvent_t event) {
    return ::musaEventSynchronize(event);
  }
};
