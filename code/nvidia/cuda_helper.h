#pragma once

#include <sstream>

#include <cuda.h>
#include <curand.h>

void throw_cuda_error(cudaError_t ret, int line, const char* filename) {
    if (ret != cudaSuccess) {
        std::stringstream ss;
        ss << "Cuda failure: " << cudaGetErrorString(ret) <<
            " in " << filename << " at line: " << line << std::endl;
        throw std::runtime_error(ss.str());
    }
}

#define CHECK_CUDA_ERROR(ret) throw_cuda_error(ret, __LINE__, __FILE__)

const char* curandGetErrorString(curandStatus_t error) {
  switch (error) {
  case CURAND_STATUS_SUCCESS:
    return "CURAND_STATUS_SUCCESS";
  case CURAND_STATUS_VERSION_MISMATCH:
    return "CURAND_STATUS_VERSION_MISMATCH";
  case CURAND_STATUS_NOT_INITIALIZED:
    return "CURAND_STATUS_NOT_INITIALIZED";
  case CURAND_STATUS_ALLOCATION_FAILED:
    return "CURAND_STATUS_ALLOCATION_FAILED";
  case CURAND_STATUS_TYPE_ERROR:
    return "CURAND_STATUS_TYPE_ERROR";
  case CURAND_STATUS_OUT_OF_RANGE:
    return "CURAND_STATUS_OUT_OF_RANGE";
  case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
    return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";
  case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
    return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";
  case CURAND_STATUS_LAUNCH_FAILURE:
    return "CURAND_STATUS_LAUNCH_FAILURE";
  case CURAND_STATUS_PREEXISTING_FAILURE:
    return "CURAND_STATUS_PREEXISTING_FAILURE";
  case CURAND_STATUS_INITIALIZATION_FAILED:
    return "CURAND_STATUS_INITIALIZATION_FAILED";
  case CURAND_STATUS_ARCH_MISMATCH:
    return "CURAND_STATUS_ARCH_MISMATCH";
  case CURAND_STATUS_INTERNAL_ERROR:
    return "CURAND_STATUS_INTERNAL_ERROR";
  default:
    return "Undefined curand status";
  }
}


void throw_curand_error(curandStatus_t ret, int line, const char* filename) {
  if (ret != CURAND_STATUS_SUCCESS) {
        std::stringstream ss;
        ss << "Curand failure: " << curandGetErrorString(ret) <<
            " in " << filename << " at line: " << line << std::endl;
	throw std::runtime_error(ss.str());
  }
}

#define CHECK_CURAND_ERROR(ret) throw_curand_error(ret, __LINE__, __FILE__)
