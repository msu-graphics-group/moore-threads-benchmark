#pragma once

#include <musa_runtime.h>


struct Musa {
  Musa() = delete;
  Musa(const Musa &) = delete;
  Musa &operator=(const Musa &) = delete;

  using cudaError_t    = musaError_t;
  using cudaEvent_t    = musaEvent_t;
  using cudaDeviceProp = musaDeviceProp;
  using cudaDeviceAttr = musaDeviceAttribute_t;
  using cudaMemcpyKind = musaMemcpyKind;
  using cudaStream_t   = musaStream_t;

  static constexpr cudaError_t cudaSuccess = musaSuccess;

  static constexpr cudaDeviceAttr cudaDevAttrClockRate                         = musaDeviceAttributeClockRate;
  static constexpr cudaDeviceAttr cudaDevAttrMemoryClockRate                   = musaDeviceAttributeMemoryClockRate;
  static constexpr cudaDeviceAttr cudaDevAttrSingleToDoublePrecisionPerfRatio  = musaDeviceAttributeSingleToDoublePrecisionPerfRatio;
  static constexpr cudaDeviceAttr cudaDevAttrMaxSharedMemoryPerMultiProcessor  = musaDeviceAttributeMaxSharedMemoryPerMultiprocessor;

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

  //--------------
  //--- Memory ---
  //--------------

  template<class T>
  static cudaError_t cudaMalloc(T **devPtr, size_t size) {
    return ::musaMalloc(devPtr, size);
  }
  
  template<class T>
  static cudaError_t cudaFree(T *devPtr) {
    return ::musaFree(devPtr);
  }

  template<class T>
  static cudaError_t cudaMemcpy(T *dst, const T *src, size_t count, cudaMemcpyKind kind) {
    return ::musaMemcpy(dst, src, count, kind);
  }

  static cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream = 0) {
    return ::musaEventRecord(event, stream);
  }

  //--------------
  //--- Events ---
  //--------------

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
