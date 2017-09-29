#pragma once
#include "cuStats.h"
#include <math.h>

#define CUSTATS_NUM_OF_THREADS 128

#define CUSTATS_BLOCK_MULTIPLIER 32

// This has to be a multiple of 2.
#define CUSTATS_MIN_SIZE_PER_THREAD 2

void cuStats_determine_launch_parameters(unsigned long int* blocks, unsigned long int* threads, unsigned long int* number_per_thread, unsigned long int max_block_size, unsigned long int max_thread_size) {
	if (*number_per_thread > CUSTATS_MIN_SIZE_PER_THREAD)
	{
		if ((*blocks * 2) <= max_block_size)
		{
			*blocks = (*blocks * 2);
			*number_per_thread = (int)ceil(*number_per_thread / 2) + 1;
			cuStats_determine_launch_parameters(blocks, threads, number_per_thread, max_block_size, max_thread_size);
		}
		else if ((*threads * 2) <= max_thread_size)
		{
			*threads = (*threads * 2);
			*number_per_thread = (int)ceil(*number_per_thread / 2) + 1;
			cuStats_determine_launch_parameters(blocks, threads, number_per_thread, max_block_size, max_thread_size);
		}
		return;
	}
	else if(*number_per_thread > 1) {
		// Because this is a multiple of two, blocks * threads * number_per_thread is usually twice the amount needed. Minus one to correct it.
		*number_per_thread = *number_per_thread - 1;
	}
	return;
}

__global__ void cuStats_standard_deviation_kernel(float *std, float *sample, unsigned long int number_per_thread, unsigned long long int data_set_size, float mean) {
	int xid = (threadIdx.x + (blockIdx.x * blockDim.x));
	if (xid + number_per_thread < data_set_size) {
		for (int i = 0; i < number_per_thread; i++) {
			std[xid + i] = powf(sample[xid + i] - mean, 2);
		}
	}
}

__global__ void cuStats_standard_deviation_kernel(double *std, double *sample, unsigned long int number_per_thread, unsigned long long int data_set_size, double mean) {
	int xid = (threadIdx.x + (blockIdx.x * blockDim.x));
	if (xid + number_per_thread < data_set_size) {
		for (int i = 0; i < number_per_thread; i++) {
			std[xid + i] = pow(sample[xid + i] - mean, 2);
		}
	}
}

template<typename T> double cuStats_standard_deviation(unsigned int device_id, T *data_set, unsigned long long int data_set_size, double mean) {
	cudaError_t gpu_device = cudaSetDevice(device_id);

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, device_id);

	unsigned long int blocks = 2;
	unsigned long int threads = CUSTATS_NUM_OF_THREADS;
	unsigned long int number_per_thread = data_set_size / (blocks * threads) + 1;

	cuStats_determine_launch_parameters(&blocks, &threads, &number_per_thread, prop.maxGridSize[0], prop.maxThreadsDim[0]);

	T *h_result = (T *)malloc(data_set_size * sizeof(T));
	memcpy(h_result, data_set, data_set_size * sizeof(T));

	T *d_a, *d_result;

	cudaMalloc(&d_a, sizeof(T) * data_set_size);
	cudaMalloc(&d_result, sizeof(T) * data_set_size);

	cudaMemcpy(d_a, data_set, sizeof(T) * data_set_size, cudaMemcpyHostToDevice);

	cuStats_standard_deviation_kernel << <blocks, threads >> > (d_result, d_a, number_per_thread, data_set_size, mean);

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

	unsigned long int blocks = 2;
	unsigned long int threads = CUSTATS_NUM_OF_THREADS;
	unsigned long int number_per_thread = data_set_size / (blocks * threads) + 1;

	cuStats_determine_launch_parameters(&blocks, &threads, &number_per_thread, prop.maxGridSize[0], prop.maxThreadsDim[0]);

	T *h_result = (T *)malloc(data_set_size * sizeof(T));
	memcpy(h_result, data_set, data_set_size * sizeof(T));

	T *d_a, *d_result;

	cudaMalloc(&d_a, sizeof(T) * data_set_size);
	cudaMalloc(&d_result, sizeof(T) * data_set_size);

	cudaMemcpy(d_a, data_set, sizeof(T) * data_set_size, cudaMemcpyHostToDevice);

	cuStats_standard_deviation_kernel << <blocks, threads >> > (d_result, d_a, number_per_thread, data_set_size, mean);

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

__global__ void cuStats_covariance_kernel(double *result, double *x_array, double x_mean, double *y_array, double y_mean, unsigned long int number_per_thread, unsigned long long int data_set_size) {
	int xid = (threadIdx.x + (blockIdx.x * blockDim.x));
	if (xid + number_per_thread < data_set_size) {
		for (int i = 0; i < number_per_thread; i++) {
			result[xid + i] = (x_array[i] - x_mean) * (y_array[i] - y_mean);
		}
	}
}

__global__ void cuStats_covariance_kernel(float *result, float *x_array, float x_mean, float *y_array, float y_mean, unsigned long int number_per_thread, unsigned long long int data_set_size) {
	int xid = (threadIdx.x + (blockIdx.x * blockDim.x));
	if (xid + number_per_thread < data_set_size) {
		for (int i = 0; i < number_per_thread; i++) {
			result[xid + i] = (x_array[i] - x_mean) * (y_array[i] - y_mean);
		}
	}
}

template<typename T> double cuStats_sample_covariance(unsigned int device_id, T *x_array, double x_mean, T *y_array, double y_mean, unsigned long long int array_size) {
	cudaError_t gpu_device = cudaSetDevice(device_id);

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, device_id);

	unsigned long int blocks = 2;
	unsigned long int threads = CUSTATS_NUM_OF_THREADS;
	unsigned long int number_per_thread = array_size / (blocks * threads) + 1;

	cuStats_determine_launch_parameters(&blocks, &threads, &number_per_thread, prop.maxGridSize[0], prop.maxThreadsDim[0]);

	T *h_result = (T *)malloc(array_size * sizeof(T));
	T *d_x, *d_y, *d_result;

	cudaMalloc(&d_x, sizeof(T) * array_size);
	cudaMalloc(&d_y, sizeof(T) * array_size);
	cudaMalloc(&d_result, sizeof(T) * array_size);

	cudaMemcpy(d_x, x_array, sizeof(T) * array_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, y_array, sizeof(T) * array_size, cudaMemcpyHostToDevice);

	cuStats_covariance_kernel << <blocks, threads >> > (d_result, d_x, x_mean, d_y, y_mean, number_per_thread, array_size);

	cudaMemcpy(h_result, d_result, sizeof(T) * array_size, cudaMemcpyDeviceToHost);

	double sum = 0;
	for (unsigned long long int j = 0; j < array_size; j++) {
		sum += h_result[j];
	}

	cudaFree(d_x);
	cudaFree(d_y);
	cudaFree(d_result);
	free(h_result);

	return ((double)1 / (array_size - 1)) * sum;
}

extern "C" __declspec(dllexport) double SampleCovarianceFloat(unsigned int device_id, float *x_array, double x_mean, float *y_array, double y_mean, unsigned long long int array_size) {
	return cuStats_sample_covariance<float>(device_id, x_array, x_mean, y_array, y_mean, array_size);
}

extern "C" __declspec(dllexport) double SampleCovarianceDouble(unsigned int device_id, double *x_array, double x_mean, double *y_array, double y_mean, unsigned long long int array_size) {
	return cuStats_sample_covariance<double>(device_id, x_array, x_mean, y_array, y_mean, array_size);
}

template<typename T> double cuStats_covariance(unsigned int device_id, T *x_array, double x_mean, T *y_array, double y_mean, unsigned long long int array_size) {
	cudaError_t gpu_device = cudaSetDevice(device_id);

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, device_id);

	unsigned long int blocks = 2;
	unsigned long int threads = CUSTATS_NUM_OF_THREADS;
	unsigned long int number_per_thread = array_size / (blocks * threads) + 1;

	cuStats_determine_launch_parameters(&blocks, &threads, &number_per_thread, prop.maxGridSize[0], prop.maxThreadsDim[0]);

	T *h_result = (T *)malloc(array_size * sizeof(T));
	T *d_x, *d_y, *d_result;

	cudaMalloc(&d_x, sizeof(T) * array_size);
	cudaMalloc(&d_y, sizeof(T) * array_size);
	cudaMalloc(&d_result, sizeof(T) * array_size);

	cudaMemcpy(d_x, x_array, sizeof(T) * array_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, y_array, sizeof(T) * array_size, cudaMemcpyHostToDevice);

	cuStats_covariance_kernel << <blocks, threads >> > (d_result, d_x, x_mean, d_y, y_mean, number_per_thread, array_size);

	cudaMemcpy(h_result, d_result, sizeof(T) * array_size, cudaMemcpyDeviceToHost);

	double sum = 0;
	for (unsigned long long int j = 0; j < array_size; j++) {
		sum += h_result[j];
	}

	cudaFree(d_x);
	cudaFree(d_y);
	cudaFree(d_result);
	free(h_result);

	return ((double)1 / array_size) * sum;
}

extern "C" __declspec(dllexport) double CovarianceFloat(unsigned int device_id, float *x_array, double x_mean, float *y_array, double y_mean, unsigned long long int array_size) {
	return cuStats_covariance<float>(device_id, x_array, x_mean, y_array, y_mean, array_size);
}

extern "C" __declspec(dllexport) double CovarianceDouble(unsigned int device_id, double *x_array, double x_mean, double *y_array, double y_mean, unsigned long long int array_size) {
	return cuStats_covariance<double>(device_id, x_array, x_mean, y_array, y_mean, array_size);
}

template<typename T> double cuStats_pearson_correlation(unsigned int device_id, T *x_array, double x_mean, T *y_array, double y_mean, unsigned long long int array_size) {
	double covariance = cuStats_covariance(device_id, x_array, x_mean, y_array, y_mean, array_size);
	double x_standard_deviation = cuStats_standard_deviation(device_id, x_array, array_size, x_mean);
	double y_standard_deviation = cuStats_standard_deviation(device_id, y_array, array_size, y_mean);
	return covariance / (x_standard_deviation * y_standard_deviation);
}

extern "C" __declspec(dllexport) double PearsonCorrelationFloat(unsigned int device_id, float *x_array, double x_mean, float *y_array, double y_mean, unsigned long long int array_size) {
	return cuStats_pearson_correlation<float>(device_id, x_array, x_mean, y_array, y_mean, array_size);
}

extern "C" __declspec(dllexport) double PearsonCorrelationDouble(unsigned int device_id, double *x_array, double x_mean, double *y_array, double y_mean, unsigned long long int array_size) {
	return cuStats_pearson_correlation<double>(device_id, x_array, x_mean, y_array, y_mean, array_size);
}
