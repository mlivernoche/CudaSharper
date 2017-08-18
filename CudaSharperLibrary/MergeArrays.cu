#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_profiler_api.h"

__global__ void copyMergeArrays(int *result, int *input, const unsigned int offset, const unsigned int length)
{
	int xid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (xid < length) {
		result[xid + offset] = input[xid];
	}
}

__global__ void copyMergeArrays(long *result, long *input, const unsigned int offset, const unsigned int length)
{
	int xid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (xid < length) {
		result[xid + offset] = input[xid];
	}
}

__global__ void copyMergeArrays(float *result, float *input, const unsigned int offset, const unsigned int length)
{
	int xid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (xid < length) {
		result[xid + offset] = input[xid];
	}
}

__global__ void copyMergeArrays(double *result, double *input, const unsigned int offset, const unsigned int length)
{
	int xid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (xid < length) {
		result[xid + offset] = input[xid];
	}
}

template<typename T> void mergeArrays(int device_id, T *result, T *array1, T *array2, const unsigned int array1_length, const unsigned int array2_length)
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

	cudaStream_t stream1;
	cudaStreamCreate(&stream1);

	cudaMemcpyAsync(d_array1, array1, array1_length * sizeof(T), cudaMemcpyHostToDevice, stream1);
	cudaMemcpyAsync(d_array2, array2, array2_length * sizeof(T), cudaMemcpyHostToDevice, stream1);

	cudaStreamSynchronize(stream1);
	cudaStreamDestroy(stream1);

	copyMergeArrays << <blocks, threadsPerBlock >> > (dResult, d_array1, 0, array1_length - 1);
	copyMergeArrays << <blocks, threadsPerBlock >> > (dResult, d_array2, array1_length, result_length);

	cudaMemcpy(result, dResult, result_length * sizeof(T), cudaMemcpyDeviceToHost);

	cudaFree(dResult);
	cudaFree(d_array1);
	cudaFree(d_array2);
}

extern "C" __declspec(dllexport) void MergeIntArrays(int device_id, int *result, int *array1, int *array2, const unsigned int array1_length, const unsigned int array2_length) {
	mergeArrays<int>(device_id, result, array1, array2, array1_length, array2_length);
}

extern "C" __declspec(dllexport) void MergeLongArrays(int device_id, long *result, long *array1, long *array2, const unsigned int array1_length, const unsigned int array2_length) {
	mergeArrays<long>(device_id, result, array1, array2, array1_length, array2_length);
}

extern "C" __declspec(dllexport) void MergeFloatArrays(int device_id, float *result, float *array1, float *array2, const unsigned int array1_length, const unsigned int array2_length) {
	mergeArrays<float>(device_id, result, array1, array2, array1_length, array2_length);
}

extern "C" __declspec(dllexport) void MergeDoubleArrays(int device_id, double *result, double *array1, double *array2, const unsigned int array1_length, const unsigned int array2_length) {
	mergeArrays<double>(device_id, result, array1, array2, array1_length, array2_length);
}
