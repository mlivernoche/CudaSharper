#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_profiler_api.h"

__global__ void copyArrays(int *result, int *input, const unsigned int offset, const unsigned int length)
{
	int xid = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (xid < length) {
		result[xid] = input[xid + offset];
	}
}

template<typename T> void splitArray(int device_id, int *src, int *array1, int *array2, const unsigned int array_length, const unsigned int split_index)
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

	cudaStream_t stream1;
	cudaStreamCreate(&stream1);

	cudaMemcpyAsync(d_src, src, array_length * sizeof(T), cudaMemcpyHostToDevice, stream1);
	cudaMemcpyAsync(d_array1, array1, array1_length * sizeof(T), cudaMemcpyHostToDevice, stream1);
	cudaMemcpyAsync(d_array2, array2, array2_length * sizeof(T), cudaMemcpyHostToDevice, stream1);

	cudaStreamSynchronize(stream1);

	cudaStream_t stream2;
	cudaStreamCreate(&stream2);

	copyArrays << <blocks, threadsPerBlock, 0, stream1 >> > (d_array1, d_src, 0, array1_length);
	copyArrays << <blocks, threadsPerBlock, 0, stream2 >> > (d_array2, d_src, split_index, array2_length);

	cudaStreamSynchronize(stream1);
	cudaStreamSynchronize(stream2);

	cudaMemcpyAsync(array1, d_array1, array1_length * sizeof(T), cudaMemcpyDeviceToHost, stream1);
	cudaMemcpyAsync(array2, d_array2, array2_length * sizeof(T), cudaMemcpyDeviceToHost, stream1);

	cudaStreamSynchronize(stream1);
	cudaStreamDestroy(stream1);
	cudaStreamDestroy(stream2);

	cudaFree(d_array1);
	cudaFree(d_array2);
}

extern "C" __declspec(dllexport) void SplitIntArray(int device_id, int *src, int *array1, int *array2, const unsigned int array_length, const unsigned int split_index);

void SplitIntArray(int device_id, int *src, int *array1, int *array2, const unsigned int array_length, const unsigned int split_index) {
	splitArray<int>(device_id, src, array1, array2, array_length, split_index);
}
