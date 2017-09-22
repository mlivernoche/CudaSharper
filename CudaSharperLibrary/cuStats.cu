#pragma once
#include "cuStats.h"
#include <math.h>

#define CUARRAY_MIN_SIZE_PER_THREAD 1

void cuArray_determine_launch_parameters(unsigned long int* blocks, unsigned long int* threads, unsigned long int* number_per_thread, unsigned long int max_block_size, unsigned long int max_thread_size) {
	if (*number_per_thread > CUARRAY_MIN_SIZE_PER_THREAD)
	{
		if ((*blocks * 2) <= max_block_size)
		{
			*blocks = (*blocks * 2);
			*number_per_thread = (*number_per_thread / 2) + 1;
			cuArray_determine_launch_parameters(blocks, threads, number_per_thread, max_block_size, max_thread_size);
		}
		else if ((*threads * 2) <= max_thread_size)
		{
			*threads = (*threads * 2);
			*number_per_thread = (*number_per_thread / 2) + 1;
			cuArray_determine_launch_parameters(blocks, threads, number_per_thread, max_block_size, max_thread_size);
		}
		return;
	}
	return;
}

__global__ void cuStats_standard_deviation_kernel(float *std, float *sample, unsigned long int number_per_thread, float mean) {
	int xid = (threadIdx.x + (blockIdx.x * blockDim.x));
	for (int i = 0; i < number_per_thread; i++) {
		std[xid + i] = powf(sample[xid + i] - mean, 2);
	}
}

__global__ void cuStats_standard_deviation_kernel(double *std, double *sample, unsigned long int number_per_thread, double mean) {
	int xid = (threadIdx.x + (blockIdx.x * blockDim.x));
	for (int i = 0; i < number_per_thread; i++) {
		std[xid + i] = pow(sample[xid + i] - mean, 2);
	}
}

template<typename T> double cuStats_standard_deviation(unsigned int device_id, T *sample, unsigned long long int sample_size, double mean) {
	cudaError_t gpu_device = cudaSetDevice(device_id);

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, device_id);

	unsigned long int threads = prop.maxThreadsPerBlock;
	unsigned long int blocks = (sample_size / threads) + 1;
	unsigned long int number_per_thread = sample_size / (blocks * threads) + 1;

	cuArray_determine_launch_parameters(&blocks, &threads, &number_per_thread, prop.maxThreadsDim[0], prop.maxThreadsPerBlock);

	T *h_result = (T *)malloc(sample_size * sizeof(T));
	memcpy(h_result, sample, sample_size * sizeof(T));

	T *d_a, *d_result;

	cudaMalloc(&d_a, sizeof(T) * sample_size);
	cudaMalloc(&d_result, sizeof(T) * sample_size);

	cudaMemcpy(d_a, sample, sizeof(T) * sample_size, cudaMemcpyHostToDevice);

	cuStats_standard_deviation_kernel << <blocks, threads >> > (d_result, d_a, number_per_thread, mean);

	cudaMemcpy(h_result, d_result, sizeof(T) * sample_size, cudaMemcpyDeviceToHost);

	double sum_std = 0;
	for (int j = 0; j < sample_size; j++) {
		double com = h_result[j];
		sum_std += com;
	}

	cudaFree(d_a);
	cudaFree(d_result);
	free(h_result);
	
	return sqrt(((double)1 / (sample_size - 1)) * sum_std);
}

extern "C" __declspec(dllexport) double SampleStandardDeviationFloat(unsigned int device_id, float *sample, unsigned long long int sample_size, double mean) {
	return cuStats_standard_deviation<float>(device_id, sample, sample_size, mean);
}

extern "C" __declspec(dllexport) double SampleStandardDeviationDouble(unsigned int device_id, double *sample, unsigned long long int sample_size, double mean) {
	return cuStats_standard_deviation<double>(device_id, sample, sample_size, mean);
}
