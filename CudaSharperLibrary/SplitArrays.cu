#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_profiler_api.h"

__global__ void copyArrays(int *result, int *input, unsigned int offset, unsigned int length)
{
	int xid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (xid < length) {
		result[xid] = input[xid + offset];
	}
}

__global__ void copyArrays(long *result, long *input, unsigned int offset, unsigned int length)
{
	int xid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (xid < length) {
		result[xid] = input[xid + offset];
	}
}

__global__ void copyArrays(float *result, float *input, unsigned int offset, unsigned int length)
{
	int xid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (xid < length) {
		result[xid] = input[xid + offset];
	}
}

__global__ void copyArrays(double *result, double *input, unsigned int offset, unsigned int length)
{
	int xid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (xid < length) {
		result[xid] = input[xid + offset];
	}
}

template<typename T> void splitArray(int device_id, T *src, T *array1, T *array2, const unsigned int array_length, const unsigned int split_index)
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

	copyArrays << <blocks, threadsPerBlock, sizeof(int) >> > (d_array1, d_src, 0, array1_length);
	copyArrays << <blocks, threadsPerBlock, sizeof(int) >> > (d_array2, d_src, split_index, array_length);

	cudaMemcpy(array1, d_array1, array1_length * sizeof(T), cudaMemcpyDeviceToHost);
	cudaMemcpy(array2, d_array2, array2_length * sizeof(T), cudaMemcpyDeviceToHost);

	cudaFree(d_src);
	cudaFree(d_array1);
	cudaFree(d_array2);
}

extern "C" __declspec(dllexport) void SplitIntArray(int device_id, int *src, int *array1, int *array2, const unsigned int array_length, const unsigned int split_index) {
	splitArray<int>(device_id, src, array1, array2, array_length, split_index);
}

extern "C" __declspec(dllexport) void SplitLongArray(int device_id, long *src, long *array1, long *array2, const unsigned int array_length, const unsigned int split_index) {
	splitArray<long>(device_id, src, array1, array2, array_length, split_index);
}

extern "C" __declspec(dllexport) void SplitFloatArray(int device_id, float *src, float *array1, float *array2, const unsigned int array_length, const unsigned int split_index) {
	splitArray<float>(device_id, src, array1, array2, array_length, split_index);
}

extern "C" __declspec(dllexport) void SplitDoubleArray(int device_id, double *src, double *array1, double *array2, const unsigned int array_length, const unsigned int split_index) {
	splitArray<double>(device_id, src, array1, array2, array_length, split_index);
}
