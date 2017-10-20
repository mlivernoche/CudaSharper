#pragma once
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_profiler_api.h"
#include "device_functions.h"
#include <stdint.h>
#include <atomic>
#include <memory>

#include "cuda_mem_mgmt.h"
#include "launch_parameters.h"
#include "DeviceInfo.h"

// These kernels and functions focus on: array arithmetics (in other words, a 1xN matrix) and array operations (split, merge, etc.).
// In this case, an array is defined as a 1xN matrix, that is, a matrix with one row and a N number of columns.

class cuArray : cuda_launch_parameters {
public:
	cuArray(int32_t device_id, int64_t amount_of_numbers) {
		this->device_id = device_id;
		this->device_ptr_0 = std::make_unique<cuda_device>(device_id, amount_of_numbers);
		this->device_ptr_1 = std::make_unique<cuda_device>(device_id, amount_of_numbers);

		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, device_id);

		// These kernels like blocks over threads.
		this->max_blocks = prop.multiProcessorCount * 32;
		this->max_threads = prop.maxThreadsPerBlock;
	}

	~cuArray() {
		this->device_ptr_0.reset();
		this->device_ptr_1.reset();
	}

	int64_t max_size() const {
		return this->device_ptr_0->max_size();
	}

	template<typename T>
	cudaError_t add_arrays(
		T* __restrict dev_ptr_0,
		T* __restrict dev_ptr_1,
		T* __restrict result,
		const T* __restrict array1,
		const T* __restrict array2,
		const int64_t full_idx);

	cudaError_t add_arrays(int32_t* __restrict result, const int32_t* __restrict array1, const int32_t* __restrict array2, const int64_t full_idx) {
		return this->add_arrays(this->device_ptr_0->getu32(), this->device_ptr_1->getu32(), result, array1, array2, full_idx);
	}
	cudaError_t add_arrays(int64_t* __restrict result, const int64_t* __restrict array1, const int64_t* __restrict array2, const int64_t full_idx) {
		return this->add_arrays(this->device_ptr_0->getu64(), this->device_ptr_1->getu64(), result, array1, array2, full_idx);
	}
	cudaError_t add_arrays(float* __restrict result, const float* __restrict array1, const float* __restrict array2, const int64_t full_idx) {
		return this->add_arrays(this->device_ptr_0->getf32(), this->device_ptr_1->getf32(), result, array1, array2, full_idx);
	}
	cudaError_t add_arrays(double* __restrict result, const double* __restrict array1, const double* __restrict array2, const int64_t full_idx) {
		return this->add_arrays(this->device_ptr_0->getf64(), this->device_ptr_1->getf64(), result, array1, array2, full_idx);
	}

	template<typename T>
	cudaError_t subtract_arrays(
		T* __restrict dev_ptr_0,
		T* __restrict dev_ptr_1,
		T* __restrict result,
		const T* __restrict array1,
		const T* __restrict array2,
		const int64_t full_idx);

	cudaError_t subtract_arrays(int32_t* __restrict result, const int32_t* __restrict array1, const int32_t* __restrict array2, const int64_t full_idx) {
		return this->subtract_arrays(this->device_ptr_0->getu32(), this->device_ptr_1->getu32(), result, array1, array2, full_idx);
	}
	cudaError_t subtract_arrays(int64_t* __restrict result, const int64_t* __restrict array1, const int64_t* __restrict array2, const int64_t full_idx) {
		return this->subtract_arrays(this->device_ptr_0->getu64(), this->device_ptr_1->getu64(), result, array1, array2, full_idx);
	}
	cudaError_t subtract_arrays(float* __restrict result, const float* __restrict array1, const float* __restrict array2, const int64_t full_idx) {
		return this->subtract_arrays(this->device_ptr_0->getf32(), this->device_ptr_1->getf32(), result, array1, array2, full_idx);
	}
	cudaError_t subtract_arrays(double* __restrict result, const double* __restrict array1, const double* __restrict array2, const int64_t full_idx) {
		return this->subtract_arrays(this->device_ptr_0->getf64(), this->device_ptr_1->getf64(), result, array1, array2, full_idx);
	}

protected:
	int32_t device_id;
	std::unique_ptr<cuda_device> device_ptr_0;
	std::unique_ptr<cuda_device> device_ptr_1;

	void determine_launch_parameters(int32_t* blocks, int32_t* threads, const int64_t array_size, const int32_t max_block_size, const int32_t max_thread_size) {
		if (*blocks * *threads < array_size)
		{
			if ((*threads * 2) < max_thread_size)
			{
				*threads = (*threads * 2);
				this->determine_launch_parameters(blocks, threads, array_size, max_block_size, max_thread_size);
			}
			else if ((*blocks * 2) < max_block_size)
			{
				*blocks = (*blocks * 2);
				this->determine_launch_parameters(blocks, threads, array_size, max_block_size, max_thread_size);
			}

			return;
		}
		return;
	}
};

/*
 * Kernels and functions for adding two arrays together.
 */

__global__ void cuArray_add_arrays_kernel(int32_t* __restrict a, const int32_t* __restrict b, const int64_t array_count);
__global__ void cuArray_add_arrays_kernel(int64_t* __restrict a, const int64_t* __restrict b, const int64_t array_count);
__global__ void cuArray_add_arrays_kernel(float* __restrict a, const float* __restrict b, const int64_t array_count);
__global__ void cuArray_add_arrays_kernel(double* __restrict a, const double* __restrict b, const int64_t array_count);

/*
* Kernels and functions for subtracting two arrays together.
*/

__global__ void cuArray_subtract_arrays_kernel(int32_t* __restrict a, const int32_t* __restrict b, const int64_t array_count);
__global__ void cuArray_subtract_arrays_kernel(int64_t* __restrict a, const int64_t* __restrict b, const int64_t array_count);
__global__ void cuArray_subtract_arrays_kernel(float* __restrict a, const float* __restrict b, const int64_t array_count);
__global__ void cuArray_subtract_arrays_kernel(double* __restrict a, const double* __restrict b, const int64_t array_count);
extern "C" {
	__declspec(dllexport) cuArray* CreateArrayClass(int32_t device_id, int64_t amount_of_numbers);
	__declspec(dllexport) void DisposeArrayClass(cuArray* rand);

	__declspec(dllexport) int32_t AddIntArrays(cuArray* arr, int32_t *result, int32_t *array1, int32_t *array2, const int64_t full_idx);
	__declspec(dllexport) int32_t AddFloatArrays(cuArray* arr, float *result, float *array1, float *array2, const int64_t full_idx);
	__declspec(dllexport) int32_t AddLongArrays(cuArray* arr, int64_t *result, int64_t *array1, int64_t *array2, const int64_t full_idx);
	__declspec(dllexport) int32_t AddDoubleArrays(cuArray* arr, double *result, double *array1, double *array2, const int64_t full_idx);

	__declspec(dllexport) int32_t SubtractIntArrays(cuArray* arr, int32_t *result, int32_t *array1, int32_t *array2, const int64_t full_idx);
	__declspec(dllexport) int32_t SubtractFloatArrays(cuArray* arr, float *result, float *array1, float *array2, const int64_t full_idx);
	__declspec(dllexport) int32_t SubtractLongArrays(cuArray* arr, int64_t *result, int64_t *array1, int64_t *array2, const int64_t full_idx);
	__declspec(dllexport) int32_t SubtractDoubleArrays(cuArray* arr, double *result, double *array1, double *array2, const int64_t full_idx);
}

