#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_profiler_api.h"

__global__ void copyMergeArrays(int *result, int *input, const unsigned int offset, const unsigned int length);
__global__ void copyMergeArrays(long *result, long *input, const unsigned int offset, const unsigned int length);
__global__ void copyMergeArrays(float *result, float *input, const unsigned int offset, const unsigned int length);
__global__ void copyMergeArrays(double *result, double *input, const unsigned int offset, const unsigned int length);

template<typename T> void mergeArrays(int device_id, T *result, T *array1, T *array2, const unsigned int array1_length, const unsigned int array2_length);

extern "C" __declspec(dllexport) void MergeIntArrays(int device_id, int *result, int *array1, int *array2, const unsigned int array1_length, const unsigned int array2_length);
extern "C" __declspec(dllexport) void MergeLongArrays(int device_id, long *result, long *array1, long *array2, const unsigned int array1_length, const unsigned int array2_length);
extern "C" __declspec(dllexport) void MergeFloatArrays(int device_id, float *result, float *array1, float *array2, const unsigned int array1_length, const unsigned int array2_length);
extern "C" __declspec(dllexport) void MergeDoubleArrays(int device_id, double *result, double *array1, double *array2, const unsigned int array1_length, const unsigned int array2_length);
