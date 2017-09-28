
#include "cuArray.h"

// sizeof(curandState_t) = 48.
// Launching 64 threads, 48 * 64 = 3072; 3072 * 32 = 98304 bytes. (32 = block limit).
// 98304 is the amount of shared memory per block available on Pascal/Maxwell!
// Using the shared memory for this kernel can halve the execution time (on Pascal).
// For Kepler, we have to double the amount of threads to achieve 100% occupancy.
#define CUARRAY_NUM_OF_THREADS 128

// This has to be a multiple of 2.
#define CUARRAY_MIN_SIZE_PER_THREAD 2

void cuArray_determine_launch_parameters(unsigned long int* blocks, unsigned long int* threads, unsigned long int* number_per_thread, unsigned long int max_block_size, unsigned long int max_thread_size) {
	if (*number_per_thread > CUARRAY_MIN_SIZE_PER_THREAD)
	{
		if ((*blocks * 2) <= max_block_size)
		{
			*blocks = (*blocks * 2);
			*number_per_thread = (int)ceil(*number_per_thread / 2) + 1;
			cuArray_determine_launch_parameters(blocks, threads, number_per_thread, max_block_size, max_thread_size);
		}
		else if ((*threads * 2) <= max_thread_size)
		{
			*threads = (*threads * 2);
			*number_per_thread = (int)ceil(*number_per_thread / 2) + 1;
			cuArray_determine_launch_parameters(blocks, threads, number_per_thread, max_block_size, max_thread_size);
		}
		return;
	}
	else if (*number_per_thread > 1) {
		// Because this is a multiple of two, blocks * threads * number_per_thread is usually twice the amount needed. Minus one to correct it.
		*number_per_thread = *number_per_thread - 1;
	}
	return;
}

/*
Functions for adding two arrays together. Requires CUDA 8.0.
*/

__global__ void cuArray_add_arrays_kernel(int *result, int *a, int *b, unsigned int number_per_thread, unsigned int array_count) {
	int xid = (blockIdx.x * blockDim.x) + threadIdx.x;

	// This is the starting point of the array that this kernel is responsible for.
	int kernel_block = (xid * number_per_thread);

	if (number_per_thread + kernel_block < array_count) {
		for (int i = 0; i < number_per_thread; i++) {
			result[kernel_block + i] = a[kernel_block + i] + b[kernel_block + i];
		}
	}
	else if (kernel_block < array_count) {
		for (int i = 0; i < array_count - kernel_block; i++) {
			result[kernel_block + i] = a[kernel_block + i] + b[kernel_block + i];
		}
	}
}

__global__ void cuArray_add_arrays_kernel(float *result, float *a, float *b, unsigned int number_per_thread, unsigned int array_count) {
	int xid = (blockIdx.x * blockDim.x) + threadIdx.x;

	// This is the starting point of the array that this kernel is responsible for.
	int kernel_block = (xid * number_per_thread);

	if (number_per_thread + kernel_block < array_count) {
		for (int i = 0; i < number_per_thread; i++) {
			result[kernel_block + i] = a[kernel_block + i] + b[kernel_block + i];
		}
	}
	else if (kernel_block < array_count) {
		for (int i = 0; i < array_count - kernel_block; i++) {
			result[kernel_block + i] = a[kernel_block + i] + b[kernel_block + i];
		}
	}
}

__global__ void cuArray_add_arrays_kernel(double *result, double *a, double *b, unsigned int number_per_thread, unsigned int array_count) {
	int xid = (blockIdx.x * blockDim.x) + threadIdx.x;

	// This is the starting point of the array that this kernel is responsible for.
	int kernel_block = (xid * number_per_thread);

	if (number_per_thread + kernel_block < array_count) {
		for (int i = 0; i < number_per_thread; i++) {
			result[kernel_block + i] = a[kernel_block + i] + b[kernel_block + i];
		}
	}
	else if (kernel_block < array_count) {
		for (int i = 0; i < array_count - kernel_block; i++) {
			result[kernel_block + i] = a[kernel_block + i] + b[kernel_block + i];
		}
	}
}

__global__ void cuArray_add_arrays_kernel(long *result, long *a, long *b, unsigned int number_per_thread, unsigned int array_count) {
	int xid = (blockIdx.x * blockDim.x) + threadIdx.x;

	// This is the starting point of the array that this kernel is responsible for.
	int kernel_block = (xid * number_per_thread);

	if (number_per_thread + kernel_block < array_count) {
		for (int i = 0; i < number_per_thread; i++) {
			result[kernel_block + i] = a[kernel_block + i] + b[kernel_block + i];
		}
	}
	else if (kernel_block < array_count) {
		for (int i = 0; i < array_count - kernel_block; i++) {
			result[kernel_block + i] = a[kernel_block + i] + b[kernel_block + i];
		}
	}
}

template<typename T> void cuArray_add_arrays(unsigned int device_id, T *result, T *array1, T *array2, const int full_idx) {
	cudaError_t gpu_device = cudaSetDevice(device_id);

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, device_id);

	unsigned long int threads = CUARRAY_NUM_OF_THREADS;
	unsigned long int blocks = 2;
	unsigned long int numberPerThread = (full_idx / (blocks * threads)) + 1;

	cuArray_determine_launch_parameters(&blocks, &threads, &numberPerThread, prop.maxGridSize[0], prop.maxThreadsDim[0]);

	T *d_a, *d_b, *d_result;

	cudaMalloc(&d_a, sizeof(T) * full_idx);
	cudaMalloc(&d_b, sizeof(T) * full_idx);
	cudaMalloc(&d_result, sizeof(T) * full_idx);

	cudaMemcpy(d_a, array1, sizeof(T) * full_idx, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, array2, sizeof(T) * full_idx, cudaMemcpyHostToDevice);

	cuArray_add_arrays_kernel << <blocks, threads >> > (d_result, d_a, d_b, numberPerThread, full_idx);

	cudaMemcpy(result, d_result, sizeof(T) * full_idx, cudaMemcpyDeviceToHost);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_result);
}

extern "C" __declspec(dllexport) void AddIntArrays(unsigned int device_id, int *result, int *array1, int *array2, const int full_idx) {
	cuArray_add_arrays<int>(device_id, result, array1, array2, full_idx);
}

extern "C" __declspec(dllexport) void AddFloatArrays(unsigned int device_id, float *result, float *array1, float *array2, const int full_idx) {
	cuArray_add_arrays<float>(device_id, result, array1, array2, full_idx);
}

extern "C" __declspec(dllexport) void AddLongArrays(unsigned int device_id, long *result, long *array1, long *array2, const int full_idx) {
	cuArray_add_arrays<long>(device_id, result, array1, array2, full_idx);
}

extern "C" __declspec(dllexport) void AddDoubleArrays(unsigned int device_id, double *result, double *array1, double *array2, const int full_idx) {
	cuArray_add_arrays<double>(device_id, result, array1, array2, full_idx);
}

/*
* Kernels and functions for subtracting two arrays.
*/

__global__ void cuArray_subtract_arrays_kernel(int *result, int *a, int *b, unsigned int array_count) {
	int xid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (xid < array_count) {
		result[xid] = a[xid] - b[xid];
	}
}

__global__ void cuArray_subtract_arrays_kernel(float *result, float *a, float *b, unsigned int array_count) {
	int xid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (xid < array_count) {
		result[xid] = a[xid] - b[xid];
	}
}

__global__ void cuArray_subtract_arrays_kernel(double *result, double *a, double *b, unsigned int array_count) {
	int xid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (xid < array_count) {
		result[xid] = a[xid] - b[xid];
	}
}

__global__ void cuArray_subtract_arrays_kernel(long *result, long *a, long *b, unsigned int array_count) {
	int xid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (xid < array_count) {
		result[xid] = a[xid] - b[xid];
	}
}

template<typename T> void cuArray_subtract_arrays(unsigned int device_id, T *result, T *array1, T *array2, const int full_idx) {
	cudaError_t gpu_device = cudaSetDevice(device_id);

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, device_id);

	const int threadsPerBlock = prop.maxThreadsPerBlock;
	const int blocks = (full_idx / threadsPerBlock) + 1;

	T *d_a, *d_b, *d_result;

	cudaMalloc(&d_a, sizeof(T) * full_idx);
	cudaMalloc(&d_b, sizeof(T) * full_idx);
	cudaMalloc(&d_result, sizeof(T) * full_idx);

	cudaMemcpy(d_a, array1, sizeof(T) * full_idx, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, array2, sizeof(T) * full_idx, cudaMemcpyHostToDevice);
	cudaMemcpy(d_result, result, sizeof(T) * full_idx, cudaMemcpyHostToDevice);

	cuArray_subtract_arrays_kernel << <blocks, threadsPerBlock >> > (d_result, d_a, d_b, full_idx);

	cudaMemcpy(result, d_result, sizeof(T) * full_idx, cudaMemcpyDeviceToHost);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_result);
}

extern "C" __declspec(dllexport) void SubtractIntArrays(unsigned int device_id, int *result, int *array1, int *array2, const int full_idx) {
	cuArray_subtract_arrays<int>(device_id, result, array1, array2, full_idx);
}

extern "C" __declspec(dllexport) void SubtractFloatArrays(unsigned int device_id, float *result, float *array1, float *array2, const int full_idx) {
	cuArray_subtract_arrays<float>(device_id, result, array1, array2, full_idx);
}

extern "C" __declspec(dllexport) void SubtractLongArrays(unsigned int device_id, long *result, long *array1, long *array2, const int full_idx) {
	cuArray_subtract_arrays<long>(device_id, result, array1, array2, full_idx);
}

extern "C" __declspec(dllexport) void SubtractDoubleArrays(unsigned int device_id, double *result, double *array1, double *array2, const int full_idx) {
	cuArray_subtract_arrays<double>(device_id, result, array1, array2, full_idx);
}

/*
* Kernels and functions for merging two arrays into one.
*/

__global__ void cuArray_merge_arrays_kernel(int *result, int *input, const unsigned int offset, const unsigned int length)
{
	int xid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (xid < length) {
		result[xid + offset] = input[xid];
	}
}

__global__ void cuArray_merge_arrays_kernel(long *result, long *input, const unsigned int offset, const unsigned int length)
{
	int xid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (xid < length) {
		result[xid + offset] = input[xid];
	}
}

__global__ void cuArray_merge_arrays_kernel(float *result, float *input, const unsigned int offset, const unsigned int length)
{
	int xid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (xid < length) {
		result[xid + offset] = input[xid];
	}
}

__global__ void cuArray_merge_arrays_kernel(double *result, double *input, const unsigned int offset, const unsigned int length)
{
	int xid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (xid < length) {
		result[xid + offset] = input[xid];
	}
}

template<typename T> void cuArray_merge_arrays(unsigned int device_id, T *result, T *array1, T *array2, const unsigned int array1_length, const unsigned int array2_length)
{
	cudaError_t gpu_device = cudaSetDevice(device_id);

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, device_id);

	const int result_length = array1_length + array2_length;
	const int threadsPerBlock = prop.maxThreadsPerBlock;
	const int blocks = (result_length / threadsPerBlock) + 1;

	T *dResult, *d_array1, *d_array2;

	cudaMalloc(&dResult, result_length * sizeof(T));
	cudaMalloc(&d_array1, array1_length * sizeof(T));
	cudaMalloc(&d_array2, array2_length * sizeof(T));

	cudaMemcpy(d_array1, array1, array1_length * sizeof(T), cudaMemcpyHostToDevice);
	cudaMemcpy(d_array2, array2, array2_length * sizeof(T), cudaMemcpyHostToDevice);

	cuArray_merge_arrays_kernel << <blocks, threadsPerBlock >> > (dResult, d_array1, 0, array1_length - 1);
	cuArray_merge_arrays_kernel << <blocks, threadsPerBlock >> > (dResult, d_array2, array1_length, result_length);

	cudaMemcpy(result, dResult, result_length * sizeof(T), cudaMemcpyDeviceToHost);

	cudaFree(dResult);
	cudaFree(d_array1);
	cudaFree(d_array2);
}

extern "C" __declspec(dllexport) void MergeIntArrays(unsigned int device_id, int *result, int *array1, int *array2, const unsigned int array1_length, const unsigned int array2_length) {
	cuArray_merge_arrays<int>(device_id, result, array1, array2, array1_length, array2_length);
}

extern "C" __declspec(dllexport) void MergeLongArrays(unsigned int device_id, long *result, long *array1, long *array2, const unsigned int array1_length, const unsigned int array2_length) {
	cuArray_merge_arrays<long>(device_id, result, array1, array2, array1_length, array2_length);
}

extern "C" __declspec(dllexport) void MergeFloatArrays(unsigned int device_id, float *result, float *array1, float *array2, const unsigned int array1_length, const unsigned int array2_length) {
	cuArray_merge_arrays<float>(device_id, result, array1, array2, array1_length, array2_length);
}

extern "C" __declspec(dllexport) void MergeDoubleArrays(unsigned int device_id, double *result, double *array1, double *array2, const unsigned int array1_length, const unsigned int array2_length) {
	cuArray_merge_arrays<double>(device_id, result, array1, array2, array1_length, array2_length);
}

/*
* Kernels and functions for splitting an array into two.
*/

__global__ void cuArray_split_arrays_kernel(int *result, int *input, unsigned int offset, unsigned int length)
{
	int xid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (xid < length) {
		result[xid] = input[xid + offset];
	}
}

__global__ void cuArray_split_arrays_kernel(long *result, long *input, unsigned int offset, unsigned int length)
{
	int xid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (xid < length) {
		result[xid] = input[xid + offset];
	}
}

__global__ void cuArray_split_arrays_kernel(float *result, float *input, unsigned int offset, unsigned int length)
{
	int xid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (xid < length) {
		result[xid] = input[xid + offset];
	}
}

__global__ void cuArray_split_arrays_kernel(double *result, double *input, unsigned int offset, unsigned int length)
{
	int xid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (xid < length) {
		result[xid] = input[xid + offset];
	}
}

template<typename T> void cuArray_split_arrays(unsigned int device_id, T *src, T *array1, T *array2, const unsigned int array_length, const unsigned int split_index)
{
	cudaError_t gpu_device = cudaSetDevice(device_id);

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, device_id);

	const int threadsPerBlock = prop.maxThreadsPerBlock;
	const int blocks = (array_length / threadsPerBlock) + 1;

	const int array1_length = split_index, array2_length = array_length - split_index;

	T *d_src, *d_array1, *d_array2;

	cudaMalloc(&d_src, array_length * sizeof(T));
	cudaMalloc(&d_array1, array1_length * sizeof(T));
	cudaMalloc(&d_array2, array2_length * sizeof(T));

	cudaMemcpy(d_src, src, array_length * sizeof(T), cudaMemcpyHostToDevice);
	cudaMemcpy(d_array1, array1, array1_length * sizeof(T), cudaMemcpyHostToDevice);
	cudaMemcpy(d_array2, array2, array2_length * sizeof(T), cudaMemcpyHostToDevice);

	cuArray_split_arrays_kernel << <blocks, threadsPerBlock, sizeof(int) >> > (d_array1, d_src, 0, array1_length);
	cuArray_split_arrays_kernel << <blocks, threadsPerBlock, sizeof(int) >> > (d_array2, d_src, split_index, array_length);

	cudaMemcpy(array1, d_array1, array1_length * sizeof(T), cudaMemcpyDeviceToHost);
	cudaMemcpy(array2, d_array2, array2_length * sizeof(T), cudaMemcpyDeviceToHost);

	cudaFree(d_src);
	cudaFree(d_array1);
	cudaFree(d_array2);
}

extern "C" __declspec(dllexport) void SplitIntArray(unsigned int device_id, int *src, int *array1, int *array2, const unsigned int array_length, const unsigned int split_index) {
	cuArray_split_arrays<int>(device_id, src, array1, array2, array_length, split_index);
}

extern "C" __declspec(dllexport) void SplitLongArray(unsigned int device_id, long *src, long *array1, long *array2, const unsigned int array_length, const unsigned int split_index) {
	cuArray_split_arrays<long>(device_id, src, array1, array2, array_length, split_index);
}

extern "C" __declspec(dllexport) void SplitFloatArray(unsigned int device_id, float *src, float *array1, float *array2, const unsigned int array_length, const unsigned int split_index) {
	cuArray_split_arrays<float>(device_id, src, array1, array2, array_length, split_index);
}

extern "C" __declspec(dllexport) void SplitDoubleArray(unsigned int device_id, double *src, double *array1, double *array2, const unsigned int array_length, const unsigned int split_index) {
	cuArray_split_arrays<double>(device_id, src, array1, array2, array_length, split_index);
}
