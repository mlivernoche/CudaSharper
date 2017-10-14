
#include "cuArray.h"

#define CUARRAY_NUM_OF_THREADS 64

#define CUARRAY_MIN_NUM_OF_BLOCKS 4

// This has to be a multiple of 2.
#define CUARRAY_MIN_SIZE_PER_THREAD 2

void cuArray::determine_launch_parameters(int32_t* blocks, int32_t* threads, const int64_t array_size, const int32_t max_block_size, const int32_t max_thread_size) {
	if (*blocks * *threads < array_size)
	{
		if ((*threads * 2) <= max_thread_size)
		{
			*threads = (*threads * 2);
			cuArray::determine_launch_parameters(blocks, threads, array_size, max_block_size, max_thread_size);
		}
		else if ((*blocks * 2) <= max_block_size)
		{
			*blocks = (*blocks * 2);
			cuArray::determine_launch_parameters(blocks, threads, array_size, max_block_size, max_thread_size);
		}
		
		return;
	}
	return;
}

/*
Functions for adding two arrays together. Requires CUDA 8.0.
*/

__global__ void cuArray_add_arrays_kernel(int32_t *a, int32_t *b, const int64_t array_count) {
	for (int i = threadIdx.x + (blockIdx.x * blockDim.x); i < array_count; i += blockDim.x * gridDim.x) {
		a[i] += b[i];
	}
}

__global__ void cuArray_add_arrays_kernel(float *a, float *b, const int64_t array_count) {
	for (int i = threadIdx.x + (blockIdx.x * blockDim.x); i < array_count; i += blockDim.x * gridDim.x) {
		a[i] += b[i];
	}
}

__global__ void cuArray_add_arrays_kernel(double *a, double *b, const int64_t array_count) {
	for (int i = threadIdx.x + (blockIdx.x * blockDim.x); i < array_count; i += blockDim.x * gridDim.x) {
		a[i] += b[i];
	}
}

__global__ void cuArray_add_arrays_kernel(int64_t *a, int64_t *b, const int64_t array_count) {
	for (int i = threadIdx.x + (blockIdx.x * blockDim.x); i < array_count; i += blockDim.x * gridDim.x) {
		a[i] += b[i];
	}
}

template<typename T> cudaError_t cuArray::add_arrays(int32_t device_id, T *result, T *array1, T *array2, const int64_t full_idx) {
	cudaError_t errorCode = cudaSetDevice(device_id);
	if (errorCode != cudaSuccess) return errorCode;

	cudaDeviceProp prop;
	errorCode = cudaGetDeviceProperties(&prop, device_id);
	if (errorCode != cudaSuccess) return errorCode;

	const size_t data_size_in_memory = sizeof(T) * full_idx;
	int32_t threads = CUARRAY_NUM_OF_THREADS;
	int32_t blocks = CUARRAY_MIN_NUM_OF_BLOCKS;
	cuArray::determine_launch_parameters(&blocks, &threads, full_idx, prop.multiProcessorCount * 32, prop.maxThreadsDim[0]);

	T *d_a, *d_b;

	errorCode = cudaMalloc(&d_a, data_size_in_memory);
	if (errorCode != cudaSuccess) return errorCode;
	errorCode = cudaMalloc(&d_b, data_size_in_memory);
	if (errorCode != cudaSuccess) return errorCode;

	errorCode = cudaMemcpy(d_a, array1, data_size_in_memory, cudaMemcpyHostToDevice);
	if (errorCode != cudaSuccess) return errorCode;
	errorCode = cudaMemcpy(d_b, array2, data_size_in_memory, cudaMemcpyHostToDevice);
	if (errorCode != cudaSuccess) return errorCode;

	cuArray_add_arrays_kernel << <blocks, threads >> > (d_a, d_b, full_idx);

	errorCode = cudaMemcpy(result, d_a, data_size_in_memory, cudaMemcpyDeviceToHost);
	if (errorCode != cudaSuccess) return errorCode;

	errorCode = cudaFree(d_a);
	if (errorCode != cudaSuccess) return errorCode;
	errorCode = cudaFree(d_b);
	if (errorCode != cudaSuccess) return errorCode;

	return cudaSuccess;
}

extern "C" {
	__declspec(dllexport) int AddIntArrays(int32_t device_id, int32_t *result, int32_t *array1, int32_t *array2, const int64_t full_idx) {
		return marshal_cuda_error(cuArray::add_arrays<int32_t>(device_id, result, array1, array2, full_idx));
	}
	__declspec(dllexport) int AddFloatArrays(int32_t device_id, float *result, float *array1, float *array2, const int64_t full_idx) {
		return marshal_cuda_error(cuArray::add_arrays<float>(device_id, result, array1, array2, full_idx));
	}
	__declspec(dllexport) int AddLongArrays(int32_t device_id, int64_t *result, int64_t *array1, int64_t *array2, const int64_t full_idx) {
		return marshal_cuda_error(cuArray::add_arrays<int64_t>(device_id, result, array1, array2, full_idx));
	}
	__declspec(dllexport) int AddDoubleArrays(int32_t device_id, double *result, double *array1, double *array2, const int64_t full_idx) {
		return marshal_cuda_error(cuArray::add_arrays<double>(device_id, result, array1, array2, full_idx));
	}
}

/*
* Kernels and functions for subtracting two arrays.
*/

__global__ void cuArray_subtract_arrays_kernel(int32_t *a, int32_t *b, const int64_t array_count) {
	for (int i = threadIdx.x + (blockIdx.x * blockDim.x); i < array_count; i += blockDim.x * gridDim.x) {
		a[i] -= b[i];
	}
}

__global__ void cuArray_subtract_arrays_kernel(float *a, float *b, const int64_t array_count) {
	for (int i = threadIdx.x + (blockIdx.x * blockDim.x); i < array_count; i += blockDim.x * gridDim.x) {
		a[i] -= b[i];
	}
}

__global__ void cuArray_subtract_arrays_kernel(double *a, double *b, const int64_t array_count) {
	for (int i = threadIdx.x + (blockIdx.x * blockDim.x); i < array_count; i += blockDim.x * gridDim.x) {
		a[i] -= b[i];
	}
}

__global__ void cuArray_subtract_arrays_kernel(int64_t *a, int64_t *b, const int64_t array_count) {
	for (int i = threadIdx.x + (blockIdx.x * blockDim.x); i < array_count; i += blockDim.x * gridDim.x) {
		a[i] -= b[i];
	}
}

template<typename T> cudaError_t cuArray::subtract_arrays(int32_t device_id, T *result, T *array1, T *array2, const int64_t full_idx) {
	cudaError_t errorCode = cudaSetDevice(device_id);
	if (errorCode != cudaSuccess) return errorCode;

	cudaDeviceProp prop;
	errorCode = cudaGetDeviceProperties(&prop, device_id);
	if (errorCode != cudaSuccess) return errorCode;

	const size_t data_size_in_memory = sizeof(T) * full_idx;
	int32_t threads = CUARRAY_NUM_OF_THREADS;
	int32_t blocks = CUARRAY_MIN_NUM_OF_BLOCKS;
	cuArray::determine_launch_parameters(&blocks, &threads, full_idx, prop.multiProcessorCount * 32, prop.maxThreadsDim[0]);

	T *d_a, *d_b;

	errorCode = cudaMalloc(&d_a, data_size_in_memory);
	if (errorCode != cudaSuccess) return errorCode;
	errorCode = cudaMalloc(&d_b, data_size_in_memory);
	if (errorCode != cudaSuccess) return errorCode;

	errorCode = cudaMemcpy(d_a, array1, data_size_in_memory, cudaMemcpyHostToDevice);
	if (errorCode != cudaSuccess) return errorCode;
	errorCode = cudaMemcpy(d_b, array2, data_size_in_memory, cudaMemcpyHostToDevice);
	if (errorCode != cudaSuccess) return errorCode;

	cuArray_subtract_arrays_kernel << <blocks, threads >> > (d_a, d_b, full_idx);

	errorCode = cudaMemcpy(result, d_a, data_size_in_memory, cudaMemcpyDeviceToHost);
	if (errorCode != cudaSuccess) return errorCode;

	errorCode = cudaFree(d_a);
	if (errorCode != cudaSuccess) return errorCode;
	errorCode = cudaFree(d_b);
	if (errorCode != cudaSuccess) return errorCode;

	return cudaSuccess;
}

extern "C" {
	__declspec(dllexport) int32_t SubtractIntArrays(int32_t device_id, int32_t *result, int32_t *array1, int32_t *array2, const int64_t full_idx) {
		return marshal_cuda_error(cuArray::subtract_arrays<int32_t>(device_id, result, array1, array2, full_idx));
	}
	__declspec(dllexport) int32_t SubtractFloatArrays(int32_t device_id, float *result, float *array1, float *array2, const int64_t full_idx) {
		return marshal_cuda_error(cuArray::subtract_arrays<float>(device_id, result, array1, array2, full_idx));
	}
	__declspec(dllexport) int32_t SubtractLongArrays(int32_t device_id, int64_t *result, int64_t *array1, int64_t *array2, const int64_t full_idx) {
		return marshal_cuda_error(cuArray::subtract_arrays<int64_t>(device_id, result, array1, array2, full_idx));
	}
	__declspec(dllexport) int32_t SubtractDoubleArrays(int32_t device_id, double *result, double *array1, double *array2, const int64_t full_idx) {
		return marshal_cuda_error(cuArray::subtract_arrays<double>(device_id, result, array1, array2, full_idx));
	}
}
