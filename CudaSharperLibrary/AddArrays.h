#pragma once
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_profiler_api.h"
#include "device_functions.h"

__global__ void addArraysKernel(int *result, int *a, int *b, unsigned int array_count);
__global__ void addArraysKernel(float *result, float *a, float *b, unsigned int array_count);
__global__ void addArraysKernel(double *result, double *a, double *b, unsigned int array_count);
__global__ void addArraysKernel(long *result, long *a, long *b, unsigned int array_count);

template<typename T> void addArrays(int device_id, T *result, T *array1, T *array2, const int full_idx);
extern "C" __declspec(dllexport) void AddIntArrays(int device_id, int *result, int *array1, int *array2, const int full_idx);
extern "C" __declspec(dllexport) void AddFloatArrays(int device_id, float *result, float *array1, float *array2, const int full_idx);
extern "C" __declspec(dllexport) void AddLongArrays(int device_id, long *result, long *array1, long *array2, const int full_idx);
extern "C" __declspec(dllexport) void AddDoubleArrays(int device_id, double *result, double *array1, double *array2, const int full_idx);
