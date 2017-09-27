#pragma once
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_profiler_api.h"
#include "device_functions.h"

#include <math.h>

// These kernels and functions focus on: array arithmetics (in other words, a 1xN matrix) and array operations (split, merge, etc.).
// In this case, an array is defined as a 1xN matrix, that is, a matrix with one row and a N number of columns.

void cuArray_determine_launch_parameters(unsigned long int* blocks, unsigned long int* threads, unsigned long int* number_per_thread, unsigned long int max_block_size, unsigned long int max_thread_size);

/*
 * Kernels and functions for adding two arrays together.
 */

__global__ void cuArray_add_arrays_kernel(int *result, int *a, int *b, unsigned int number_per_thread, unsigned int array_count);
__global__ void cuArray_add_arrays_kernel(float *result, float *a, float *b, unsigned int number_per_thread, unsigned int array_count);
__global__ void cuArray_add_arrays_kernel(double *result, double *a, double *b, unsigned int number_per_thread, unsigned int array_count);
__global__ void cuArray_add_arrays_kernel(long *result, long *a, long *b, unsigned int number_per_thread, unsigned int array_count);
template<typename T> void cuArray_add_arrays(unsigned int device_id, T *result, T *array1, T *array2, const int full_idx);
extern "C" __declspec(dllexport) void AddIntArrays(unsigned int device_id, int *result, int *array1, int *array2, const int full_idx);
extern "C" __declspec(dllexport) void AddFloatArrays(unsigned int device_id, float *result, float *array1, float *array2, const int full_idx);
extern "C" __declspec(dllexport) void AddLongArrays(unsigned int device_id, long *result, long *array1, long *array2, const int full_idx);
extern "C" __declspec(dllexport) void AddDoubleArrays(unsigned int device_id, double *result, double *array1, double *array2, const int full_idx);

/*
* Kernels and functions for subtracting two arrays together.
*/

__global__ void cuArray_subtract_arrays_kernel(int *result, int *a, int *b, unsigned int array_count);
__global__ void cuArray_subtract_arrays_kernel(float *result, float *a, float *b, unsigned int array_count);
__global__ void cuArray_subtract_arrays_kernel(double *result, double *a, double *b, unsigned int array_count);
__global__ void cuArray_subtract_arrays_kernel(long *result, long *a, long *b, unsigned int array_count);
template<typename T> void cuArray_subtract_arrays(unsigned int device_id, T *result, T *array1, T *array2, const int full_idx);
extern "C" __declspec(dllexport) void SubtractIntArrays(unsigned int device_id, int *result, int *array1, int *array2, const int full_idx);
extern "C" __declspec(dllexport) void SubtractFloatArrays(unsigned int device_id, float *result, float *array1, float *array2, const int full_idx);
extern "C" __declspec(dllexport) void SubtractLongArrays(unsigned int device_id, long *result, long *array1, long *array2, const int full_idx);
extern "C" __declspec(dllexport) void SubtractDoubleArrays(unsigned int device_id, double *result, double *array1, double *array2, const int full_idx);

/*
 * Kernels and functions for merging two arrays together.
 */

__global__ void cuArray_merge_arrays_kernel(int *result, int *input, const unsigned int offset, const unsigned int length);
__global__ void cuArray_merge_arrays_kernel(long *result, long *input, const unsigned int offset, const unsigned int length);
__global__ void cuArray_merge_arrays_kernel(float *result, float *input, const unsigned int offset, const unsigned int length);
__global__ void cuArray_merge_arrays_kernel(double *result, double *input, const unsigned int offset, const unsigned int length);
template<typename T> void cuArray_merge_arrays(unsigned int device_id, T *result, T *array1, T *array2, const unsigned int array1_length, const unsigned int array2_length);
extern "C" __declspec(dllexport) void MergeIntArrays(unsigned int device_id, int *result, int *array1, int *array2, const unsigned int array1_length, const unsigned int array2_length);
extern "C" __declspec(dllexport) void MergeLongArrays(unsigned int device_id, long *result, long *array1, long *array2, const unsigned int array1_length, const unsigned int array2_length);
extern "C" __declspec(dllexport) void MergeFloatArrays(unsigned int device_id, float *result, float *array1, float *array2, const unsigned int array1_length, const unsigned int array2_length);
extern "C" __declspec(dllexport) void MergeDoubleArrays(unsigned int device_id, double *result, double *array1, double *array2, const unsigned int array1_length, const unsigned int array2_length);

/*
 * Kernels and functions for splitting an array into two.
 */

__global__ void cuArray_split_arrays_kernel(int *result, int *input, unsigned int offset, unsigned int length);
__global__ void cuArray_split_arrays_kernel(long *result, long *input, unsigned int offset, unsigned int length);
__global__ void cuArray_split_arrays_kernel(float *result, float *input, unsigned int offset, unsigned int length);
__global__ void cuArray_split_arrays_kernel(double *result, double *input, unsigned int offset, unsigned int length);
template<typename T> void cuArray_split_arrays(unsigned int device_id, T *src, T *array1, T *array2, const unsigned int array_length, const unsigned int split_index);
extern "C" __declspec(dllexport) void SplitIntArray(unsigned int device_id, int *src, int *array1, int *array2, const unsigned int array_length, const unsigned int split_index);
extern "C" __declspec(dllexport) void SplitLongArray(unsigned int device_id, long *src, long *array1, long *array2, const unsigned int array_length, const unsigned int split_index);
extern "C" __declspec(dllexport) void SplitFloatArray(unsigned int device_id, float *src, float *array1, float *array2, const unsigned int array_length, const unsigned int split_index);
extern "C" __declspec(dllexport) void SplitDoubleArray(unsigned int device_id, double *src, double *array1, double *array2, const unsigned int array_length, const unsigned int split_index);

