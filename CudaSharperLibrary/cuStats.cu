#pragma once
#include "cuStats.h"

#define CUSTATS_NUM_OF_THREADS 64

#define CUSTATS_MIN_NUM_OF_BLOCKS 2

void cuStats::determine_launch_parameters(int32_t* __restrict blocks, int32_t* __restrict threads, const int64_t array_size, const int32_t max_block_size, const int32_t max_thread_size) {
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

__global__ void cuStats_standard_deviation_kernel(float* __restrict std, const float* __restrict sample, const int64_t data_set_size, const float mean) {
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	float4 std_data;

#pragma unroll
	for (int i = idx; i < data_set_size / 4; i += blockDim.x * gridDim.x) {
		std_data.x = (sample[i] - mean) * (sample[i] - mean);
		std_data.y = (sample[i + 1] - mean) * (sample[i + 1] - mean);
		std_data.z = (sample[i + 2] - mean) * (sample[i + 2] - mean);
		std_data.w = (sample[i + 3] - mean) * (sample[i + 3] - mean);
		reinterpret_cast<float4*>(std)[i] = std_data;
	}

#pragma unroll
	for (int i = idx + (data_set_size / 4) * 4; i < data_set_size; i += idx) {
		std[i] = (sample[i] - mean) * (sample[i] - mean);
	}
}

__global__ void cuStats_standard_deviation_kernel(double* __restrict std, const double* __restrict sample, const int64_t data_set_size, const double mean) {
	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	double4 std_data;

#pragma unroll
	for (int i = idx; i < data_set_size / 4; i += blockDim.x * gridDim.x) {
		std_data.x = (sample[i] - mean) * (sample[i] - mean);
		std_data.y = (sample[i + 1] - mean) * (sample[i + 1] - mean);
		std_data.z = (sample[i + 2] - mean) * (sample[i + 2] - mean);
		std_data.w = (sample[i + 3] - mean) * (sample[i + 3] - mean);
		reinterpret_cast<double4*>(std)[i] = std_data;
	}

#pragma unroll
	for (int i = idx + (data_set_size / 4) * 4; i < data_set_size; i += idx) {
		std[i] = (sample[i] - mean) * (sample[i] - mean);
	}
}

template<typename T> cudaError_t cuStats::standard_deviation_summation(const int32_t device_id, double& __restrict result, const T* __restrict sample, const int64_t sample_size, const T mean) {
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

	//size_t shared_memory = sizeof(T) * 4 * threads;
	cuStats_standard_deviation_kernel << <blocks, threads, 0 >> > (d_result, d_a, sample_size, mean);

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

template<typename T> cudaError_t cuStats::standard_deviation(const int32_t device_id, double& __restrict result, const T* __restrict sample, const int64_t sample_size, const T mean) {
	double sum = 0;
	cudaError_t errorCode = cuStats::standard_deviation_summation(device_id, sum, sample, sample_size, mean);
	result = sqrt(sum / sample_size);
	return errorCode;
}

template<typename T> cudaError_t cuStats::sample_standard_deviation(const int32_t device_id, double& __restrict result, const T* __restrict sample, const int64_t sample_size, const T mean) {
	double sum = 0;
	cudaError_t errorCode = cuStats::standard_deviation_summation(device_id, sum, sample, sample_size, mean);
	result = sqrt(sum / (sample_size - 1));
	return errorCode;
}

__global__ void cuStats_covariance_kernel(float* __restrict result, const float* __restrict x_array, const float x_mean, const float* __restrict y_array, const float y_mean, const int64_t data_set_size) {
	int idx = threadIdx.x + (blockIdx.x * blockDim.x);
	float4 cov_f32;

#pragma unroll
	for (int i = idx; i < data_set_size / 4; i += blockDim.x * gridDim.x) {
		cov_f32.x = (x_array[i] - x_mean) * (y_array[i] - y_mean);
		cov_f32.y = (x_array[i + 1] - x_mean) * (y_array[i + 1] - y_mean);
		cov_f32.z = (x_array[i + 2] - x_mean) * (y_array[i + 2] - y_mean);
		cov_f32.w = (x_array[i + 3] - x_mean) * (y_array[i + 3] - y_mean);
		reinterpret_cast<float4*>(result)[i] = cov_f32;
	}

#pragma unroll
	for (int i = idx + (data_set_size / 4) * 4; i < data_set_size; i += idx) {
		result[i] = (x_array[i] - x_mean) * (y_array[i] - y_mean);
	}
}

__global__ void cuStats_covariance_kernel(double* __restrict result, const double* __restrict x_array, const double x_mean, const double* __restrict y_array, const double y_mean, const int64_t data_set_size) {
	int idx = threadIdx.x + (blockIdx.x * blockDim.x);
	double4 cov_f64;

#pragma unroll
	for (int i = idx; i < data_set_size / 4; i += blockDim.x * gridDim.x) {
		cov_f64.x = (x_array[i] - x_mean) * (y_array[i] - y_mean);
		cov_f64.y = (x_array[i + 1] - x_mean) * (y_array[i + 1] - y_mean);
		cov_f64.z = (x_array[i + 2] - x_mean) * (y_array[i + 2] - y_mean);
		cov_f64.w = (x_array[i + 3] - x_mean) * (y_array[i + 3] - y_mean);
		reinterpret_cast<double4*>(result)[i] = cov_f64;
	}

#pragma unroll
	for (int i = idx + (data_set_size / 4) * 4; i < data_set_size; i += idx) {
		result[i] = (x_array[i] - x_mean) * (y_array[i] - y_mean);
	}
}

template<typename T> cudaError_t cuStats::covariance_summation(const int32_t device_id, double& __restrict result, const T* __restrict x_array, const T x_mean, const T* __restrict y_array, const T y_mean, const int64_t array_size) {
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
	T* h_result = (T*)malloc(data_size_in_memory);
	T* d_x, * d_y, * d_result;

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

	cuStats_covariance_kernel << <blocks, threads, 0 >> > (d_result, d_x, x_mean, d_y, y_mean, array_size);

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

template<typename T> cudaError_t cuStats::covariance(const int32_t device_id, double& __restrict result, const T* __restrict x_array, const T x_mean, const T* __restrict y_array, const T y_mean, const int64_t array_size) {
	double sum = 0;
	cudaError_t errorCode = cuStats::covariance_summation(device_id, sum, x_array, x_mean, y_array, y_mean, array_size);
	if (errorCode != cudaSuccess) return errorCode;
	result = sum / array_size;
	return errorCode;
}

template<typename T> cudaError_t cuStats::sample_covariance(const int32_t device_id, double& __restrict result, const T* __restrict x_array, const T x_mean, const T* __restrict y_array, const T y_mean, const int64_t array_size) {
	double sum = 0;
	cudaError_t errorCode = cuStats::covariance_summation(device_id, sum, x_array, x_mean, y_array, y_mean, array_size);
	if (errorCode != cudaSuccess) return errorCode;
	result = sum / (array_size - 1);
	return errorCode;
}

template<typename T> cudaError_t cuStats::pearson_correlation(const int32_t device_id, double& __restrict result, const T* __restrict x_array, const T x_mean, const T* __restrict y_array, const T y_mean, const int64_t array_size) {
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
