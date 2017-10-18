#pragma once
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_profiler_api.h"
#include "device_functions.h"
#include <stdint.h>
#include <math.h>

#include "DeviceInfo.h"

// These kernels and functions focus on: array arithmetics (in other words, a 1xN matrix) and array operations (split, merge, etc.).
// In this case, an array is defined as a 1xN matrix, that is, a matrix with one row and a N number of columns.

class cuArray {
public:
	template<typename T> static cudaError_t add_arrays(const int32_t device_id, T* __restrict result, const T* __restrict array1, const T* __restrict array2, const int64_t full_idx);
	template<typename T> static cudaError_t subtract_arrays(const int32_t device_id, T* __restrict result, const T* __restrict array1, const T* __restrict array2, const int64_t full_idx);

protected:
	static void determine_launch_parameters(int32_t* blocks, int32_t* threads, const int64_t array_size, const int32_t max_block_size, const int32_t max_thread_size);
};

/*
 * Kernels and functions for adding two arrays together.
 */

__global__ void cuArray_add_arrays_kernel(int32_t* __restrict a, const int32_t* __restrict b, const int64_t array_count);
__global__ void cuArray_add_arrays_kernel(int64_t* __restrict a, const int64_t* __restrict b, const int64_t array_count);
__global__ void cuArray_add_arrays_kernel(float* __restrict a, const float* __restrict b, const int64_t array_count);
__global__ void cuArray_add_arrays_kernel(double* __restrict a, const double* __restrict b, const int64_t array_count);
extern "C" {
	__declspec(dllexport) int32_t AddIntArrays(int32_t device_id, int32_t *result, int32_t *array1, int32_t *array2, const int64_t full_idx);
	__declspec(dllexport) int32_t AddFloatArrays(int32_t device_id, float *result, float *array1, float *array2, const int64_t full_idx);
	__declspec(dllexport) int32_t AddLongArrays(int32_t device_id, int64_t *result, int64_t *array1, int64_t *array2, const int64_t full_idx);
	__declspec(dllexport) int32_t AddDoubleArrays(int32_t device_id, double *result, double *array1, double *array2, const int64_t full_idx);
}

/*
* Kernels and functions for subtracting two arrays together.
*/

__global__ void cuArray_subtract_arrays_kernel(int32_t* __restrict a, const int32_t* __restrict b, const int64_t array_count);
__global__ void cuArray_subtract_arrays_kernel(int64_t* __restrict a, const int64_t* __restrict b, const int64_t array_count);
__global__ void cuArray_subtract_arrays_kernel(float* __restrict a, const float* __restrict b, const int64_t array_count);
__global__ void cuArray_subtract_arrays_kernel(double* __restrict a, const double* __restrict b, const int64_t array_count);
extern "C" {
	__declspec(dllexport) int32_t SubtractIntArrays(int32_t device_id, int32_t *result, int32_t *array1, int32_t *array2, const int64_t full_idx);
	__declspec(dllexport) int32_t SubtractFloatArrays(int32_t device_id, float *result, float *array1, float *array2, const int64_t full_idx);
	__declspec(dllexport) int32_t SubtractLongArrays(int32_t device_id, int64_t *result, int64_t *array1, int64_t *array2, const int64_t full_idx);
	__declspec(dllexport) int32_t SubtractDoubleArrays(int32_t device_id, double *result, double *array1, double *array2, const int64_t full_idx);
}

