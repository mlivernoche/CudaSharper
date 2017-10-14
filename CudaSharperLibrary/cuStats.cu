#pragma once
#include "cuStats.h"

#define CUSTATS_NUM_OF_THREADS 64

#define CUSTATS_MIN_NUM_OF_BLOCKS 2

void cuStats::determine_launch_parameters(int32_t* blocks, int32_t* threads, const int64_t array_size, const int32_t max_block_size, const int32_t max_thread_size) {
	if (*blocks * *threads < array_size) {
		if ((*threads * 2) <= max_thread_size)
		{
			*threads = (*threads * 2);
			cuStats::determine_launch_parameters(blocks, threads, array_size, max_block_size, max_thread_size);
		}
		else if ((*blocks * 2) <= max_block_size)
		{
			*blocks = (*blocks * 2);
			cuStats::determine_launch_parameters(blocks, threads, array_size, max_block_size, max_thread_size);
		}
		return;
	}
	return;
}

__global__ void cuStats_standard_deviation_kernel(float* std, float* sample, const int64_t data_set_size, float mean) {
	extern __shared__ float4 std_f32[];

	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	float4 std_data;

	for (int i = idx; i < data_set_size / 4; i += blockDim.x * gridDim.x) {
		std_f32[threadIdx.x].x = sample[i];
		std_f32[threadIdx.x].y = sample[i + 1];
		std_f32[threadIdx.x].z = sample[i + 2];
		std_f32[threadIdx.x].w = sample[i + 3];

		std_data.x = (std_f32[threadIdx.x].x - mean) * (std_f32[threadIdx.x].x - mean);
		std_data.y = (std_f32[threadIdx.x].y - mean) * (std_f32[threadIdx.x].y - mean);
		std_data.z = (std_f32[threadIdx.x].z - mean) * (std_f32[threadIdx.x].z - mean);
		std_data.w = (std_f32[threadIdx.x].w - mean) * (std_f32[threadIdx.x].w - mean);
		reinterpret_cast<float4*>(std)[i] = std_data;
	}

	for (int i = idx + (data_set_size / 4) * 4; i < data_set_size; i += idx) {
		std[i] = (sample[i] - mean) * (sample[i] - mean);
	}
}

__global__ void cuStats_standard_deviation_kernel(double* std, double* sample, const int64_t data_set_size, double mean) {
	extern __shared__ double4 std_f64[];

	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	double4 std_data;

	for (int i = idx; i < data_set_size / 4; i += blockDim.x * gridDim.x) {
		std_f64[threadIdx.x].x = sample[i];
		std_f64[threadIdx.x].y = sample[i + 1];
		std_f64[threadIdx.x].z = sample[i + 2];
		std_f64[threadIdx.x].w = sample[i + 3];

		std_data.x = (std_f64[threadIdx.x].x - mean) * (std_f64[threadIdx.x].x - mean);
		std_data.y = (std_f64[threadIdx.x].y - mean) * (std_f64[threadIdx.x].y - mean);
		std_data.z = (std_f64[threadIdx.x].z - mean) * (std_f64[threadIdx.x].z - mean);
		std_data.w = (std_f64[threadIdx.x].w - mean) * (std_f64[threadIdx.x].w - mean);
		reinterpret_cast<double4*>(std)[i] = std_data;
	}

	for (int i = idx + (data_set_size / 4) * 4; i < data_set_size; i += idx) {
		std[i] = (sample[i] - mean) * (sample[i] - mean);
	}
}

template<typename T> cudaError_t cuStats::standard_deviation_summation(int32_t device_id, double &result, T *sample, const int64_t sample_size, T mean) {
	cudaError_t errorCode = cudaSetDevice(device_id);
	if (errorCode != cudaSuccess) return errorCode;

	cudaDeviceProp prop;
	errorCode = cudaGetDeviceProperties(&prop, device_id);
	if (errorCode != cudaSuccess) return errorCode;

	const size_t data_size_in_memory = sizeof(T) * sample_size;
	int32_t blocks = CUSTATS_MIN_NUM_OF_BLOCKS;
	int32_t threads = CUSTATS_NUM_OF_THREADS;

	cuStats::determine_launch_parameters(&blocks, &threads, sample_size, prop.multiProcessorCount * 32, prop.maxThreadsDim[0]);

	// cudaHostAlloc() allows faster copying from DtoH, but it is much slower than malloc().
	// The benefit does not outweigh the cost.
	T *h_result = (T*)malloc(data_size_in_memory);
	T *d_a, *d_result;

	errorCode = cudaMalloc(&d_a, data_size_in_memory);
	if (errorCode != cudaSuccess) return errorCode;
	errorCode = cudaMalloc(&d_result, data_size_in_memory);
	if (errorCode != cudaSuccess) return errorCode;

	errorCode = cudaMemcpy(d_a, sample, data_size_in_memory, cudaMemcpyHostToDevice);
	if (errorCode != cudaSuccess) return errorCode;

	size_t shared_memory = sizeof(T) * 4 * threads;
	cuStats_standard_deviation_kernel << <blocks, threads, shared_memory >> > (d_result, d_a, sample_size, mean);

	errorCode = cudaMemcpy(h_result, d_result, data_size_in_memory, cudaMemcpyDeviceToHost);
	if (errorCode != cudaSuccess) return errorCode;

	double sum_std = 0;
	for (int64_t j = 0; j < sample_size; j++) {
		sum_std += (double)h_result[j];
	}

	errorCode = cudaFree(d_a);
	if (errorCode != cudaSuccess) return errorCode;
	errorCode = cudaFree(d_result);
	if (errorCode != cudaSuccess) return errorCode;
	free(h_result);

	result = sum_std;

	return cudaSuccess;
}

template<typename T> cudaError_t cuStats::standard_deviation(int32_t device_id, double &result, T *data_set, const int64_t data_set_size, T mean) {
	double sum = 0;
	cudaError_t errorCode = cuStats::standard_deviation_summation(device_id, sum, data_set, data_set_size, mean);
	result = sqrt(sum / data_set_size);
	return errorCode;
}

template<typename T> cudaError_t cuStats::sample_standard_deviation(int32_t device_id, double &result, T *data_set, const int64_t data_set_size, T mean) {
	double sum = 0;
	cudaError_t errorCode = cuStats::standard_deviation_summation(device_id, sum, data_set, data_set_size, mean);
	result = sqrt(sum / (data_set_size - 1));
	return errorCode;
}

__global__ void cuStats_covariance_kernel(double *result, double *x_array, double x_mean, double *y_array, double y_mean, const int64_t data_set_size) {
	extern __shared__ double4 cov_f64[];

	int idx = threadIdx.x + (blockIdx.x * blockDim.x);

	for (int i = idx; i < data_set_size / 4; i += blockDim.x * gridDim.x) {
		cov_f64[threadIdx.x].x = (x_array[i] - x_mean) * (y_array[i] - y_mean);
		cov_f64[threadIdx.x].y = (x_array[i + 1] - x_mean) * (y_array[i + 1] - y_mean);
		cov_f64[threadIdx.x].z = (x_array[i + 2] - x_mean) * (y_array[i + 2] - y_mean);
		cov_f64[threadIdx.x].w = (x_array[i + 3] - x_mean) * (y_array[i + 3] - y_mean);
		reinterpret_cast<double4*>(result)[i] = cov_f64[threadIdx.x];
	}

	for (int i = idx + (data_set_size / 4) * 4; i < data_set_size; i += idx) {
		result[i] = (x_array[i] - x_mean) * (y_array[i] - y_mean);
	}
}

__global__ void cuStats_covariance_kernel(float *result, float *x_array, float x_mean, float *y_array, float y_mean, const int64_t data_set_size) {
	extern __shared__ float4 cov_f32[];

	int idx = threadIdx.x + (blockIdx.x * blockDim.x);

	for (int i = idx; i < data_set_size / 4; i += blockDim.x * gridDim.x) {
		cov_f32[threadIdx.x].x = (x_array[i] - x_mean) * (y_array[i] - y_mean);
		cov_f32[threadIdx.x].y = (x_array[i + 1] - x_mean) * (y_array[i + 1] - y_mean);
		cov_f32[threadIdx.x].z = (x_array[i + 2] - x_mean) * (y_array[i + 2] - y_mean);
		cov_f32[threadIdx.x].w = (x_array[i + 3] - x_mean) * (y_array[i + 3] - y_mean);
		reinterpret_cast<float4*>(result)[i] = cov_f32[threadIdx.x];
	}

	for (int i = idx + (data_set_size / 4) * 4; i < data_set_size; i += idx) {
		result[i] = (x_array[i] - x_mean) * (y_array[i] - y_mean);
	}
}

template<typename T> cudaError_t cuStats::covariance_summation(int32_t device_id, double &result, T *x_array, T x_mean, T *y_array, T y_mean, const int64_t array_size) {
	cudaError_t errorCode = cudaSetDevice(device_id);
	if (errorCode != cudaSuccess) return errorCode;

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, device_id);
	if (errorCode != cudaSuccess) return errorCode;

	const size_t data_size_in_memory = sizeof(T) * array_size;
	int32_t blocks = CUSTATS_MIN_NUM_OF_BLOCKS;
	int32_t threads = CUSTATS_NUM_OF_THREADS;

	cuStats::determine_launch_parameters(&blocks, &threads, array_size, prop.multiProcessorCount * 32, prop.maxThreadsDim[0]);

	// cudaHostAlloc() allows faster copying from DtoH, but it is much slower than malloc().
	// The benefit does not outweigh the cost.
	T *h_result = (T *)malloc(array_size * sizeof(T));
	T *d_x, *d_y, *d_result;

	errorCode = cudaMalloc(&d_x, data_size_in_memory);
	if (errorCode != cudaSuccess) return errorCode;
	errorCode = cudaMalloc(&d_y, data_size_in_memory);
	if (errorCode != cudaSuccess) return errorCode;
	errorCode = cudaMalloc(&d_result, data_size_in_memory);
	if (errorCode != cudaSuccess) return errorCode;

	errorCode = cudaMemcpy(d_x, x_array, data_size_in_memory, cudaMemcpyHostToDevice);
	if (errorCode != cudaSuccess) return errorCode;
	errorCode = cudaMemcpy(d_y, y_array, data_size_in_memory, cudaMemcpyHostToDevice);
	if (errorCode != cudaSuccess) return errorCode;

	size_t shared_memory = sizeof(T) * 4 * threads;
	cuStats_covariance_kernel << <blocks, threads, shared_memory >> > (d_result, d_x, x_mean, d_y, y_mean, array_size);

	errorCode = cudaMemcpyAsync(h_result, d_result, data_size_in_memory, cudaMemcpyDeviceToHost);
	if (errorCode != cudaSuccess) return errorCode;

	double sum = 0;
	for (int64_t j = 0; j < array_size; j++) {
		sum += (double)h_result[j];
	}

	errorCode = cudaFree(d_x);
	if (errorCode != cudaSuccess) return errorCode;
	errorCode = cudaFree(d_y);
	if (errorCode != cudaSuccess) return errorCode;
	errorCode = cudaFree(d_result);
	if (errorCode != cudaSuccess) return errorCode;
	free(h_result);

	result = sum;

	return cudaSuccess;
}

template<typename T> cudaError_t cuStats::covariance(int32_t device_id, double &result, T *x_array, T x_mean, T *y_array, T y_mean, const int64_t array_size) {
	double sum = 0;
	cudaError_t errorCode = cuStats::covariance_summation(device_id, sum, x_array, x_mean, y_array, y_mean, array_size);
	if (errorCode != cudaSuccess) return errorCode;
	result = sum / array_size;
	return errorCode;
}

template<typename T> cudaError_t cuStats::sample_covariance(int32_t device_id, double &result, T *x_array, T x_mean, T *y_array, T y_mean, const int64_t array_size) {
	double sum = 0;
	cudaError_t errorCode = cuStats::covariance_summation(device_id, sum, x_array, x_mean, y_array, y_mean, array_size);
	if (errorCode != cudaSuccess) return errorCode;
	result = sum / (array_size - 1);
	return errorCode;
}

template<typename T> cudaError_t cuStats::pearson_correlation(int32_t device_id, double &result, T *x_array, T x_mean, T *y_array, T y_mean, const int64_t array_size) {
	// Corr(x, y) = Cov(x, y) / (Std(x) * Std(y))
	
	double covariance = 0, x_standard_deviation = 0, y_standard_deviation = 0;

	cudaError_t errorCode = cuStats::covariance(device_id, covariance, x_array, x_mean, y_array, y_mean, array_size);
	if (errorCode != cudaSuccess) return errorCode;
	errorCode = cuStats::standard_deviation(device_id, x_standard_deviation, x_array, array_size, x_mean);
	if (errorCode != cudaSuccess) return errorCode;
	errorCode = cuStats::standard_deviation(device_id, y_standard_deviation, y_array, array_size, y_mean);
	if (errorCode != cudaSuccess) return errorCode;

	result = covariance / (x_standard_deviation * y_standard_deviation);

	return cudaSuccess;
}

extern "C" {
	__declspec(dllexport) int32_t StandardDeviationFloat(int32_t device_id, double &result, float *population, const int64_t population_size, float mean) {
		return marshal_cuda_error(cuStats::standard_deviation<float>(device_id, result, population, population_size, mean));
	}
	__declspec(dllexport) int32_t StandardDeviationDouble(int32_t device_id, double &result, double *population, const int64_t population_size, double mean) {
		return marshal_cuda_error(cuStats::standard_deviation<double>(device_id, result, population, population_size, mean));
	}

	__declspec(dllexport) int32_t SampleStandardDeviationFloat(int32_t device_id, double &result, float *sample, const int64_t sample_size, float mean) {
		return marshal_cuda_error(cuStats::sample_standard_deviation<float>(device_id, result, sample, sample_size, mean));
	}
	__declspec(dllexport) int32_t SampleStandardDeviationDouble(int32_t device_id, double &result, double *sample, const int64_t sample_size, double mean) {
		return marshal_cuda_error(cuStats::sample_standard_deviation<double>(device_id, result, sample, sample_size, mean));
	}

	__declspec(dllexport) int32_t CovarianceFloat(int32_t device_id, double &result, float *x_array, float x_mean, float *y_array, float y_mean, const int64_t array_size) {
		return marshal_cuda_error(cuStats::covariance<float>(device_id, result, x_array, x_mean, y_array, y_mean, array_size));
	}
	__declspec(dllexport) int32_t CovarianceDouble(int32_t device_id, double &result, double *x_array, double x_mean, double *y_array, double y_mean, const int64_t array_size) {
		return marshal_cuda_error(cuStats::covariance<double>(device_id, result, x_array, x_mean, y_array, y_mean, array_size));
	}
	
	__declspec(dllexport) int32_t SampleCovarianceFloat(int32_t device_id, double &result, float *x_array, float x_mean, float *y_array, float y_mean, const int64_t array_size) {
		return marshal_cuda_error(cuStats::sample_covariance<float>(device_id, result, x_array, x_mean, y_array, y_mean, array_size));
	}
	__declspec(dllexport) int32_t SampleCovarianceDouble(int32_t device_id, double &result, double *x_array, double x_mean, double *y_array, double y_mean, const int64_t array_size) {
		return marshal_cuda_error(cuStats::sample_covariance<double>(device_id, result, x_array, x_mean, y_array, y_mean, array_size));
	}

	__declspec(dllexport) int32_t PearsonCorrelationFloat(int32_t device_id, double &result, float *x_array, float x_mean, float *y_array, float y_mean, const int64_t array_size) {
		return marshal_cuda_error(cuStats::pearson_correlation<float>(device_id, result, x_array, x_mean, y_array, y_mean, array_size));
	}
	__declspec(dllexport) int32_t PearsonCorrelationDouble(int32_t device_id, double &result, double *x_array, double x_mean, double *y_array, double y_mean, const int64_t array_size) {
		return marshal_cuda_error(cuStats::pearson_correlation<double>(device_id, result, x_array, x_mean, y_array, y_mean, array_size));
	}
}
