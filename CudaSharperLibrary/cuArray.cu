
#include "cuArray.h"

#define CUARRAY_NUM_OF_THREADS 64

#define CUARRAY_MIN_NUM_OF_BLOCKS 4

// This has to be a multiple of 2.
#define CUARRAY_MIN_SIZE_PER_THREAD 2

/*
Functions for adding two arrays together. Requires CUDA 8.0.
*/

__global__ void cuArray_add_arrays_kernel(int32_t* __restrict a, const int32_t* __restrict b, const int64_t array_count) {
	for (int i = threadIdx.x + (blockIdx.x * blockDim.x); i < array_count; i += blockDim.x * gridDim.x) {
		a[i] += b[i];
	}
}

__global__ void cuArray_add_arrays_kernel(int64_t* __restrict a, const int64_t* __restrict b, const int64_t array_count) {
	for (int i = threadIdx.x + (blockIdx.x * blockDim.x); i < array_count; i += blockDim.x * gridDim.x) {
		a[i] += b[i];
	}
}

__global__ void cuArray_add_arrays_kernel(float* __restrict a, const float* __restrict b, const int64_t array_count) {
	for (int i = threadIdx.x + (blockIdx.x * blockDim.x); i < array_count; i += blockDim.x * gridDim.x) {
		a[i] += b[i];
	}
}

__global__ void cuArray_add_arrays_kernel(double* __restrict a, const double* __restrict b, const int64_t array_count) {
	for (int i = threadIdx.x + (blockIdx.x * blockDim.x); i < array_count; i += blockDim.x * gridDim.x) {
		a[i] += b[i];
	}
}

template<typename T> cudaError_t cuArray::add_arrays<T>(
	T* __restrict dev_ptr_0,
	T* __restrict dev_ptr_1,
	T* __restrict result,
	const T* __restrict array1,
	const T* __restrict array2,
	const int64_t full_idx) {
	cudaError_t errorCode = cudaSetDevice(this->device_id);
	if (errorCode != cudaSuccess) return errorCode;

	const size_t data_size_in_memory = sizeof(T) * full_idx;
	int32_t threads = CUARRAY_NUM_OF_THREADS;
	int32_t blocks = CUARRAY_MIN_NUM_OF_BLOCKS;
	this->determine_launch_parameters(&blocks, &threads, full_idx, this->max_blocks, this->max_threads);

	errorCode = cudaMemcpy(dev_ptr_0, array1, data_size_in_memory, cudaMemcpyHostToDevice);
	if (errorCode != cudaSuccess) return errorCode;
	errorCode = cudaMemcpy(dev_ptr_1, array2, data_size_in_memory, cudaMemcpyHostToDevice);
	if (errorCode != cudaSuccess) return errorCode;

	cuArray_add_arrays_kernel << <blocks, threads >> > (dev_ptr_0, dev_ptr_1, full_idx);

	errorCode = cudaMemcpy(result, dev_ptr_0, data_size_in_memory, cudaMemcpyDeviceToHost);
	if (errorCode != cudaSuccess) return errorCode;

	return cudaSuccess;
}

/*
* Kernels and functions for subtracting two arrays.
*/

__global__ void cuArray_subtract_arrays_kernel(int32_t* __restrict a, const int32_t* __restrict b, const int64_t array_count) {
	for (int i = threadIdx.x + (blockIdx.x * blockDim.x); i < array_count; i += blockDim.x * gridDim.x) {
		a[i] -= b[i];
	}
}

__global__ void cuArray_subtract_arrays_kernel(int64_t* __restrict a, const int64_t* __restrict b, const int64_t array_count) {
	for (int i = threadIdx.x + (blockIdx.x * blockDim.x); i < array_count; i += blockDim.x * gridDim.x) {
		a[i] -= b[i];
	}
}

__global__ void cuArray_subtract_arrays_kernel(float* __restrict a, const float* __restrict b, const int64_t array_count) {
	for (int i = threadIdx.x + (blockIdx.x * blockDim.x); i < array_count; i += blockDim.x * gridDim.x) {
		a[i] -= b[i];
	}
}

__global__ void cuArray_subtract_arrays_kernel(double* __restrict a, const double* __restrict b, const int64_t array_count) {
	for (int i = threadIdx.x + (blockIdx.x * blockDim.x); i < array_count; i += blockDim.x * gridDim.x) {
		a[i] -= b[i];
	}
}

template<typename T> cudaError_t cuArray::subtract_arrays<T>(
	T* __restrict dev_ptr_0,
	T* __restrict dev_ptr_1,
	T* __restrict result,
	const T* __restrict array1,
	const T* __restrict array2,
	const int64_t full_idx) {
	cudaError_t errorCode = cudaSetDevice(this->device_id);
	if (errorCode != cudaSuccess) return errorCode;

	const size_t data_size_in_memory = sizeof(T) * full_idx;
	int32_t threads = CUARRAY_NUM_OF_THREADS;
	int32_t blocks = CUARRAY_MIN_NUM_OF_BLOCKS;
	this->determine_launch_parameters(&blocks, &threads, full_idx, this->max_blocks, this->max_threads);

	errorCode = cudaMemcpy(dev_ptr_0, array1, data_size_in_memory, cudaMemcpyHostToDevice);
	if (errorCode != cudaSuccess) return errorCode;
	errorCode = cudaMemcpy(dev_ptr_1, array2, data_size_in_memory, cudaMemcpyHostToDevice);
	if (errorCode != cudaSuccess) return errorCode;

	cuArray_subtract_arrays_kernel << <blocks, threads >> > (dev_ptr_0, dev_ptr_1, full_idx);

	errorCode = cudaMemcpy(result, dev_ptr_0, data_size_in_memory, cudaMemcpyDeviceToHost);
	if (errorCode != cudaSuccess) return errorCode;

	return cudaSuccess;
}

extern "C" {
	__declspec(dllexport) cuArray* CreateArrayClass(int32_t device_id, int64_t amount_of_numbers) {
		return new cuArray(device_id, amount_of_numbers);
	}
	__declspec(dllexport) void DisposeArrayClass(cuArray* arr) {
		if (arr != NULL) {
			delete arr;
			arr = NULL;
		}
	}

	__declspec(dllexport) int32_t AddIntArrays(cuArray* arr, int32_t *result, int32_t *array1, int32_t *array2, const int64_t full_idx) {
		return marshal_cuda_error(arr->add_arrays(result, array1, array2, full_idx));
	}
	__declspec(dllexport) int32_t AddFloatArrays(cuArray* arr, float *result, float *array1, float *array2, const int64_t full_idx) {
		return marshal_cuda_error(arr->add_arrays(result, array1, array2, full_idx));
	}
	__declspec(dllexport) int32_t AddLongArrays(cuArray* arr, int64_t *result, int64_t *array1, int64_t *array2, const int64_t full_idx) {
		return marshal_cuda_error(arr->add_arrays(result, array1, array2, full_idx));
	}
	__declspec(dllexport) int32_t AddDoubleArrays(cuArray* arr, double *result, double *array1, double *array2, const int64_t full_idx) {
		return marshal_cuda_error(arr->add_arrays(result, array1, array2, full_idx));
	}

	__declspec(dllexport) int32_t SubtractIntArrays(cuArray* arr, int32_t *result, int32_t *array1, int32_t *array2, const int64_t full_idx) {
		return marshal_cuda_error(arr->subtract_arrays(result, array1, array2, full_idx));
	}
	__declspec(dllexport) int32_t SubtractFloatArrays(cuArray* arr, float *result, float *array1, float *array2, const int64_t full_idx) {
		return marshal_cuda_error(arr->subtract_arrays(result, array1, array2, full_idx));
	}
	__declspec(dllexport) int32_t SubtractLongArrays(cuArray* arr, int64_t *result, int64_t *array1, int64_t *array2, const int64_t full_idx) {
		return marshal_cuda_error(arr->subtract_arrays(result, array1, array2, full_idx));
	}
	__declspec(dllexport) int32_t SubtractDoubleArrays(cuArray* arr, double *result, double *array1, double *array2, const int64_t full_idx) {
		return marshal_cuda_error(arr->subtract_arrays(result, array1, array2, full_idx));
	}
}
