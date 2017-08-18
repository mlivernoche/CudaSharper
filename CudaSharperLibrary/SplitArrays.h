#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_profiler_api.h"

__global__ void copyArrays(int *result, int *input, unsigned int offset, unsigned int length);
__global__ void copyArrays(long *result, long *input, unsigned int offset, unsigned int length);
__global__ void copyArrays(float *result, float *input, unsigned int offset, unsigned int length);
__global__ void copyArrays(double *result, double *input, unsigned int offset, unsigned int length);

template<typename T> void splitArray(int device_id, T *src, T *array1, T *array2, const unsigned int array_length, const unsigned int split_index);
extern "C" __declspec(dllexport) void SplitIntArray(int device_id, int *src, int *array1, int *array2, const unsigned int array_length, const unsigned int split_index);
extern "C" __declspec(dllexport) void SplitLongArray(int device_id, long *src, long *array1, long *array2, const unsigned int array_length, const unsigned int split_index);
extern "C" __declspec(dllexport) void SplitFloatArray(int device_id, float *src, float *array1, float *array2, const unsigned int array_length, const unsigned int split_index);
extern "C" __declspec(dllexport) void SplitDoubleArray(int device_id, double *src, double *array1, double *array2, const unsigned int array_length, const unsigned int split_index);
