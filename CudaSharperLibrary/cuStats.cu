#pragma once
#include "cuStats.h"
#include <math.h>

#define CUSTATS_NUM_OF_THREADS 128

#define CUSTATS_BLOCK_MULTIPLIER 32

#define CUSTATS_MIN_SIZE_PER_THREAD 1

void cuArray_determine_launch_parameters(unsigned long int* blocks, unsigned long int* threads, unsigned long int* number_per_thread, unsigned long int max_block_size, unsigned long int max_thread_size) {
	if (*number_per_thread > CUSTATS_MIN_SIZE_PER_THREAD)
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

template<typename T> double cuStats_standard_deviation(unsigned int device_id, T *data_set, unsigned long long int data_set_size, double mean) {
	cudaError_t gpu_device = cudaSetDevice(device_id);

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, device_id);

	unsigned long int blocks = prop.maxThreadsDim[0] * CUSTATS_BLOCK_MULTIPLIER;
	unsigned long int threads = CUSTATS_NUM_OF_THREADS;
	unsigned long int number_per_thread = data_set_size / (blocks * threads) + 1;

	cuArray_determine_launch_parameters(&blocks, &threads, &number_per_thread, prop.maxThreadsDim[0], prop.maxThreadsPerBlock);

	T *h_result = (T *)malloc(data_set_size * sizeof(T));
	memcpy(h_result, data_set, data_set_size * sizeof(T));

	T *d_a, *d_result;

	cudaMalloc(&d_a, sizeof(T) * data_set_size);
	cudaMalloc(&d_result, sizeof(T) * data_set_size);

	cudaMemcpy(d_a, data_set, sizeof(T) * data_set_size, cudaMemcpyHostToDevice);

	cuStats_standard_deviation_kernel << <blocks, threads >> > (d_result, d_a, number_per_thread, mean);

	cudaMemcpy(h_result, d_result, sizeof(T) * data_set_size, cudaMemcpyDeviceToHost);

	double sum_std = 0;
	for (int j = 0; j < data_set_size; j++) {
		sum_std += h_result[j];
	}

	cudaFree(d_a);
	cudaFree(d_result);
	free(h_result);
	
	return sqrt(((double)1 / data_set_size) * sum_std);
}

extern "C" __declspec(dllexport) double StandardDeviationFloat(unsigned int device_id, float *population, unsigned long long int population_size, double mean) {
	return cuStats_standard_deviation<float>(device_id, population, population_size, mean);
}

extern "C" __declspec(dllexport) double StandardDeviationDouble(unsigned int device_id, double *population, unsigned long long int population_size, double mean) {
	return cuStats_standard_deviation<double>(device_id, population, population_size, mean);
}

template<typename T> double cuStats_sample_standard_deviation(unsigned int device_id, T *data_set, unsigned long long int data_set_size, double mean) {
	cudaError_t gpu_device = cudaSetDevice(device_id);

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, device_id);

	unsigned long int blocks = prop.maxThreadsDim[0] * CUSTATS_BLOCK_MULTIPLIER;
	unsigned long int threads = CUSTATS_NUM_OF_THREADS;
	unsigned long int number_per_thread = data_set_size / (blocks * threads) + 1;

	cuArray_determine_launch_parameters(&blocks, &threads, &number_per_thread, prop.maxThreadsDim[0], prop.maxThreadsPerBlock);

	T *h_result = (T *)malloc(data_set_size * sizeof(T));
	memcpy(h_result, data_set, data_set_size * sizeof(T));

	T *d_a, *d_result;

	cudaMalloc(&d_a, sizeof(T) * data_set_size);
	cudaMalloc(&d_result, sizeof(T) * data_set_size);

	cudaMemcpy(d_a, data_set, sizeof(T) * data_set_size, cudaMemcpyHostToDevice);

	cuStats_standard_deviation_kernel << <blocks, threads >> > (d_result, d_a, number_per_thread, mean);

	cudaMemcpy(h_result, d_result, sizeof(T) * data_set_size, cudaMemcpyDeviceToHost);

	double sum_std = 0;
	for (int j = 0; j < data_set_size; j++) {
		sum_std += h_result[j];
	}

	cudaFree(d_a);
	cudaFree(d_result);
	free(h_result);

	return sqrt(((double)1 / (data_set_size - 1)) * sum_std);
}

extern "C" __declspec(dllexport) double SampleStandardDeviationFloat(unsigned int device_id, float *sample, unsigned long long int sample_size, double mean) {
	return cuStats_sample_standard_deviation<float>(device_id, sample, sample_size, mean);
}

extern "C" __declspec(dllexport) double SampleStandardDeviationDouble(unsigned int device_id, double *sample, unsigned long long int sample_size, double mean) {
	return cuStats_sample_standard_deviation<double>(device_id, sample, sample_size, mean);
}
