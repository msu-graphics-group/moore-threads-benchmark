#pragma once

#include <hip/hip_runtime.h>


struct Hip {
  Hip() = delete;
  Hip(const Hip &) = delete;
  Hip &operator=(const Hip &) = delete;

  using cudaError_t    = hipError_t;
  using cudaEvent_t    = hipEvent_t;
  using cudaDeviceProp = hipDeviceProp_t;
  using cudaDeviceAttr = hipDeviceAttribute_t;
  using cudaMemcpyKind = hipMemcpyKind;
  using cudaStream_t   = hipStream_t;

  static constexpr cudaError_t cudaSuccess = ::hipSuccess;

  static constexpr cudaDeviceAttr cudaDevAttrClockRate                         = hipDeviceAttributeClockRate;
  static constexpr cudaDeviceAttr cudaDevAttrMemoryClockRate                   = hipDeviceAttributeMemoryClockRate;
  static constexpr cudaDeviceAttr cudaDevAttrSingleToDoublePrecisionPerfRatio  = hipDeviceAttributeSingleToDoublePrecisionPerfRatio;
  static constexpr cudaDeviceAttr cudaDevAttrMaxSharedMemoryPerMultiProcessor  = hipDeviceAttributeMaxSharedMemoryPerMultiprocessor;

  static constexpr cudaMemcpyKind cudaMemcpyHostToDevice    = ::hipMemcpyHostToDevice;
  static constexpr cudaMemcpyKind cudaMemcpyDeviceToHost    = ::hipMemcpyDeviceToHost;
  static constexpr cudaMemcpyKind cudaMemcpyDeviceToDevice  = ::hipMemcpyDeviceToDevice;


  //--------------
  //--- Errors ---
  //--------------

  static cudaError_t cudaGetLastError() {
    return ::hipGetLastError();
  }

  static const char *cudaGetErrorString(cudaError_t error) {
    return ::hipGetErrorString(error);
  }

  //---------------
  //--- Devices ---
  //---------------

  static cudaError_t cudaDriverGetVersion(int *version) {
    return ::hipDriverGetVersion(version);
  }
  
  static cudaError_t cudaRuntimeGetVersion(int *version) {
    return ::hipRuntimeGetVersion(version);
  }

  static cudaError_t cudaGetDeviceCount(int *count) {
    return ::hipGetDeviceCount(count);
  }

  static cudaError_t cudaSetDevice(int device) {
    return ::hipSetDevice(device);
  }

  static cudaError_t cudaGetDeviceProperties(cudaDeviceProp *prop, int device) {
    return ::hipGetDeviceProperties(prop, device);
  }

  static cudaError_t cudaDeviceGetAttribute(int *value, cudaDeviceAttr attr, int device) {
    return ::hipDeviceGetAttribute(value, attr, device);
  }

  static cudaError_t cudaDeviceSynchronize() {
    return ::hipDeviceSynchronize();
  }

  //--------------
  //--- Memory ---
  //--------------

  template<class T>
  static cudaError_t cudaMalloc(T **devPtr, size_t size) {
    return ::hipMalloc(devPtr, size);
  }

  template<class T>
  static cudaError_t cudaMallocManaged(T **devPtr, size_t size) {
    return ::hipMallocManaged(devPtr, size);
  }
  
  template<class T>
  static cudaError_t cudaFree(T *devPtr) {
    return ::hipFree(devPtr);
  }

  template<class T>
  static cudaError_t cudaFreeHost(T *devPtr) {
    return ::hipFreeHost(devPtr);
  }

  template<class T>
  static cudaError_t cudaMemset(T *devPtr, int value, size_t count) {
    return ::hipMemset(devPtr, value, count);
  }

  template<class T>
  static cudaError_t cudaMemcpy(T *dst, const T *src, size_t count, cudaMemcpyKind kind) {
    return ::hipMemcpy(dst, src, count, kind);
  }

  template<class T>
  static cudaError_t cudaMemcpyAsync(T *dst, const T *src, size_t count,
                                     cudaMemcpyKind kind, cudaStream_t stream = 0) {
    return ::hipMemcpyAsync(dst, src, count, kind, stream);
  }

  static cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream = 0) {
    return ::hipEventRecord(event, stream);
  }
  
  //--------------
  //--- Events ---
  //--------------

  static cudaError_t cudaEventCreate(cudaEvent_t *event) {
    return ::hipEventCreate(event);
  }

  static cudaError_t cudaEventDestroy(cudaEvent_t event) {
    return ::hipEventDestroy(event);
  }

  static cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end) {
    return ::hipEventElapsedTime(ms, start, end);
  }

  static cudaError_t cudaEventSynchronize(cudaEvent_t event) {
    return ::hipEventSynchronize(event);
  }
};
