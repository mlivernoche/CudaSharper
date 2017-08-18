#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_profiler_api.h"
#include "device_functions.h"

/*
 Functions for adding two arrays together. Requires CUDA 8.0.
*/

__global__ void addArraysKernel(int *result, int *a, int *b, unsigned int array_count) {
	int xid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (xid < array_count) {
		result[xid] = a[xid] + b[xid];
	}
}

__global__ void addArraysKernel(float *result, float *a, float *b, unsigned int array_count) {
	int xid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (xid < array_count) {
		result[xid] = a[xid] + b[xid];
	}
}

__global__ void addArraysKernel(double *result, double *a, double *b, unsigned int array_count) {
	int xid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (xid < array_count) {
		result[xid] = a[xid] + b[xid];
	}
}

__global__ void addArraysKernel(long *result, long *a, long *b, unsigned int array_count) {
	int xid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (xid < array_count) {
		result[xid] = a[xid] + b[xid];
	}
}

template<typename T> void addArrays(int device_id, T *result, T *array1, T *array2, const int full_idx) {
	cudaError_t gpu_device = cudaSetDevice(device_id);

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, device_id);

	const int threadsPerBlock = prop.maxThreadsPerBlock;
	const int blocks = (full_idx / threadsPerBlock) + 1;

	T *d_a, *d_b, *dResult;

	cudaMalloc(&d_a, sizeof(T) * full_idx);
	cudaMalloc(&d_b, sizeof(T) * full_idx);
	cudaMalloc(&dResult, sizeof(T) * full_idx);

	cudaMemcpy(d_a, array1, sizeof(T) * full_idx, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, array2, sizeof(T) * full_idx, cudaMemcpyHostToDevice);
	cudaMemcpy(dResult, result, sizeof(T) * full_idx, cudaMemcpyHostToDevice);

	addArraysKernel << <blocks, threadsPerBlock >> > (dResult, d_a, d_b, full_idx);

	cudaMemcpy(result, dResult, sizeof(T) * full_idx, cudaMemcpyDeviceToHost);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(dResult);
}

extern "C" __declspec(dllexport) void AddIntArrays(int device_id, int *result, int *array1, int *array2, const int full_idx) {
	addArrays<int>(device_id, result, array1, array2, full_idx);
}

extern "C" __declspec(dllexport) void AddFloatArrays(int device_id, float *result, float *array1, float *array2, const int full_idx) {
	addArrays<float>(device_id, result, array1, array2, full_idx);
}

extern "C" __declspec(dllexport) void AddLongArrays(int device_id, long *result, long *array1, long *array2, const int full_idx) {
	addArrays<long>(device_id, result, array1, array2, full_idx);
}

extern "C" __declspec(dllexport) void AddDoubleArrays(int device_id, double *result, double *array1, double *array2, const int full_idx) {
	addArrays<double>(device_id, result, array1, array2, full_idx);
}

