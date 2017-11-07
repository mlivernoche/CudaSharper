#pragma once
#include "cuArray.h"

namespace csl {
	
	cudaError_t cuArray::add_arrays(int32_t* __restrict result, const int32_t* __restrict array1, const int32_t* __restrict array2, const int64_t full_idx) {
		return this->add_arrays(this->device_ptr_0->getu32(), this->device_ptr_1->getu32(), result, array1, array2, full_idx);
	}
	cudaError_t cuArray::add_arrays(int64_t* __restrict result, const int64_t* __restrict array1, const int64_t* __restrict array2, const int64_t full_idx) {
		return this->add_arrays(this->device_ptr_0->getu64(), this->device_ptr_1->getu64(), result, array1, array2, full_idx);
	}
	cudaError_t cuArray::add_arrays(float* __restrict result, const float* __restrict array1, const float* __restrict array2, const int64_t full_idx) {
		return this->add_arrays(this->device_ptr_0->getf32(), this->device_ptr_1->getf32(), result, array1, array2, full_idx);
	}
	cudaError_t cuArray::add_arrays(double* __restrict result, const double* __restrict array1, const double* __restrict array2, const int64_t full_idx) {
		return this->add_arrays(this->device_ptr_0->getf64(), this->device_ptr_1->getf64(), result, array1, array2, full_idx);
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

		kernels::cuArray::add_arrays_kernel << <blocks, threads >> > (dev_ptr_0, dev_ptr_1, full_idx);

		errorCode = cudaMemcpy(result, dev_ptr_0, data_size_in_memory, cudaMemcpyDeviceToHost);
		if (errorCode != cudaSuccess) return errorCode;

		return cudaSuccess;
	}

	cudaError_t cuArray::subtract_arrays(int32_t* __restrict result, const int32_t* __restrict array1, const int32_t* __restrict array2, const int64_t full_idx) {
		return this->subtract_arrays(this->device_ptr_0->getu32(), this->device_ptr_1->getu32(), result, array1, array2, full_idx);
	}
	cudaError_t cuArray::subtract_arrays(int64_t* __restrict result, const int64_t* __restrict array1, const int64_t* __restrict array2, const int64_t full_idx) {
		return this->subtract_arrays(this->device_ptr_0->getu64(), this->device_ptr_1->getu64(), result, array1, array2, full_idx);
	}
	cudaError_t cuArray::subtract_arrays(float* __restrict result, const float* __restrict array1, const float* __restrict array2, const int64_t full_idx) {
		return this->subtract_arrays(this->device_ptr_0->getf32(), this->device_ptr_1->getf32(), result, array1, array2, full_idx);
	}
	cudaError_t cuArray::subtract_arrays(double* __restrict result, const double* __restrict array1, const double* __restrict array2, const int64_t full_idx) {
		return this->subtract_arrays(this->device_ptr_0->getf64(), this->device_ptr_1->getf64(), result, array1, array2, full_idx);
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

		kernels::cuArray::subtract_arrays_kernel << <blocks, threads >> > (dev_ptr_0, dev_ptr_1, full_idx);

		errorCode = cudaMemcpy(result, dev_ptr_0, data_size_in_memory, cudaMemcpyDeviceToHost);
		if (errorCode != cudaSuccess) return errorCode;

		return cudaSuccess;
	}

	namespace kernels {
		namespace cuArray {
			__global__ void add_arrays_kernel(int32_t* __restrict a, const int32_t* __restrict b, const int64_t array_count) {
				for (int i = threadIdx.x + (blockIdx.x * blockDim.x); i < array_count; i += blockDim.x * gridDim.x) {
					a[i] += b[i];
				}
			}
			__global__ void add_arrays_kernel(int64_t* __restrict a, const int64_t* __restrict b, const int64_t array_count) {
				for (int i = threadIdx.x + (blockIdx.x * blockDim.x); i < array_count; i += blockDim.x * gridDim.x) {
					a[i] += b[i];
				}
			}
			__global__ void add_arrays_kernel(float* __restrict a, const float* __restrict b, const int64_t array_count) {
				for (int i = threadIdx.x + (blockIdx.x * blockDim.x); i < array_count; i += blockDim.x * gridDim.x) {
					a[i] += b[i];
				}
			}
			__global__ void add_arrays_kernel(double* __restrict a, const double* __restrict b, const int64_t array_count) {
				for (int i = threadIdx.x + (blockIdx.x * blockDim.x); i < array_count; i += blockDim.x * gridDim.x) {
					a[i] += b[i];
				}
			}

			__global__ void subtract_arrays_kernel(int32_t* __restrict a, const int32_t* __restrict b, const int64_t array_count) {
				for (int i = threadIdx.x + (blockIdx.x * blockDim.x); i < array_count; i += blockDim.x * gridDim.x) {
					a[i] -= b[i];
				}
			}
			__global__ void subtract_arrays_kernel(int64_t* __restrict a, const int64_t* __restrict b, const int64_t array_count) {
				for (int i = threadIdx.x + (blockIdx.x * blockDim.x); i < array_count; i += blockDim.x * gridDim.x) {
					a[i] -= b[i];
				}
			}
			__global__ void subtract_arrays_kernel(float* __restrict a, const float* __restrict b, const int64_t array_count) {
				for (int i = threadIdx.x + (blockIdx.x * blockDim.x); i < array_count; i += blockDim.x * gridDim.x) {
					a[i] -= b[i];
				}
			}
			__global__ void subtract_arrays_kernel(double* __restrict a, const double* __restrict b, const int64_t array_count) {
				for (int i = threadIdx.x + (blockIdx.x * blockDim.x); i < array_count; i += blockDim.x * gridDim.x) {
					a[i] -= b[i];
				}
			}
		}
	}
}
