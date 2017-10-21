#pragma once
#include "cuStats.h"

#define CUSTATS_NUM_OF_THREADS 64

#define CUSTATS_MIN_NUM_OF_BLOCKS 2

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

template<typename T>
cudaError_t cuStats::standard_deviation_summation<T>(
	T* __restrict result_ptr,
	T* __restrict sample_ptr,
	double& __restrict result,
	const T* __restrict sample, const int64_t sample_size,
	const T mean) {
	cudaError_t errorCode = cudaSetDevice(this->device_id);
	if (errorCode != cudaSuccess) return errorCode;

	const size_t data_size_in_memory = sizeof(T) * sample_size;
	int32_t blocks = CUSTATS_MIN_NUM_OF_BLOCKS;
	int32_t threads = CUSTATS_NUM_OF_THREADS;
	this->determine_launch_parameters(&blocks, &threads, sample_size, this->max_blocks, this->max_threads);

	errorCode = cudaMemcpy(sample_ptr, sample, data_size_in_memory, cudaMemcpyHostToDevice);
	if (errorCode != cudaSuccess) return errorCode;

	cuStats_standard_deviation_kernel << <blocks, threads >> > (result_ptr, sample_ptr, sample_size, mean);

	thrust::device_ptr<T> wrapped_ptr = thrust::device_pointer_cast(result_ptr);
	result = (double)thrust::reduce(thrust::device, wrapped_ptr, wrapped_ptr + sample_size, (T)0, thrust::plus<T>());

	return cudaSuccess;
}

template<typename T> cudaError_t cuStats::standard_deviation<T>(double& __restrict result, const T* __restrict sample, const int64_t sample_size, const T mean) {
	double sum = 0;
	cudaError_t errorCode = this->standard_deviation_summation(sum, sample, sample_size, mean);
	result = sqrt(sum / sample_size);
	return errorCode;
}

template<typename T> cudaError_t cuStats::sample_standard_deviation<T>(double& __restrict result, const T* __restrict sample, const int64_t sample_size, const T mean) {
	double sum = 0;
	cudaError_t errorCode = this->standard_deviation_summation(sum, sample, sample_size, mean);
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

template<typename T> cudaError_t cuStats::covariance_summation<T>(
	T* __restrict result_ptr,
	T* __restrict x_ptr,
	T* __restrict y_ptr,
	double& __restrict result,
	const T* __restrict x_array, const T x_mean,
	const T* __restrict y_array, const T y_mean,
	const int64_t array_size) {
	cudaError_t errorCode = cudaSetDevice(this->device_id);
	if (errorCode != cudaSuccess) return errorCode;

	const size_t data_size_in_memory = sizeof(T) * array_size;
	int32_t blocks = CUSTATS_MIN_NUM_OF_BLOCKS;
	int32_t threads = CUSTATS_NUM_OF_THREADS;
	this->determine_launch_parameters(&blocks, &threads, array_size, this->max_blocks, this->max_threads);

	errorCode = cudaMemcpy(x_ptr, x_array, data_size_in_memory, cudaMemcpyHostToDevice);
	if (errorCode != cudaSuccess) return errorCode;
	errorCode = cudaMemcpy(y_ptr, y_array, data_size_in_memory, cudaMemcpyHostToDevice);
	if (errorCode != cudaSuccess) return errorCode;

	cuStats_covariance_kernel << <blocks, threads, 0 >> > (result_ptr, x_ptr, x_mean, y_ptr, y_mean, array_size);

	thrust::device_ptr<T> wrapped_ptr = thrust::device_pointer_cast(result_ptr);
	result = (double)thrust::reduce(thrust::device, wrapped_ptr, wrapped_ptr + array_size, (T)0, thrust::plus<T>());

	return cudaSuccess;
}

template<typename T> cudaError_t cuStats::covariance<T>(double& __restrict result, const T* __restrict x_array, const T x_mean, const T* __restrict y_array, const T y_mean, const int64_t array_size) {
	double sum = 0;
	cudaError_t errorCode = this->covariance_summation(sum, x_array, x_mean, y_array, y_mean, array_size);
	if (errorCode != cudaSuccess) return errorCode;
	result = sum / array_size;
	return errorCode;
}

template<typename T> cudaError_t cuStats::sample_covariance<T>(double& __restrict result, const T* __restrict x_array, const T x_mean, const T* __restrict y_array, const T y_mean, const int64_t array_size) {
	double sum = 0;
	cudaError_t errorCode = this->covariance_summation(sum, x_array, x_mean, y_array, y_mean, array_size);
	if (errorCode != cudaSuccess) return errorCode;
	result = sum / (array_size - 1);
	return errorCode;
}

template<typename T> cudaError_t cuStats::pearson_correlation<T>(double& __restrict result, const T* __restrict x_array, const T x_mean, const T* __restrict y_array, const T y_mean, const int64_t array_size) {
	// Corr(x, y) = Cov(x, y) / (Std(x) * Std(y))

	double covariance = 0, x_standard_deviation = 0, y_standard_deviation = 0;

	cudaError_t errorCode = this->covariance(covariance, x_array, x_mean, y_array, y_mean, array_size);
	if (errorCode != cudaSuccess) return errorCode;
	errorCode = this->standard_deviation(x_standard_deviation, x_array, array_size, x_mean);
	if (errorCode != cudaSuccess) return errorCode;
	errorCode = this->standard_deviation(y_standard_deviation, y_array, array_size, y_mean);
	if (errorCode != cudaSuccess) return errorCode;

	result = covariance / (x_standard_deviation * y_standard_deviation);

	return cudaSuccess;
}

extern "C" {
	__declspec(dllexport) cuStats* CreateStatClass(int32_t device_id, int64_t amount_of_numbers) {
		return new cuStats(device_id, amount_of_numbers);
	}
	__declspec(dllexport) void DisposeStatClass(cuStats* stat) {
		if (stat != NULL) {
			delete stat;
			stat = NULL;
		}
	}

	__declspec(dllexport) int32_t StandardDeviationFloat(cuStats* stat, double &result, float *population, const int64_t population_size, float mean) {
		return marshal_cuda_error(stat->standard_deviation(result, population, population_size, mean));
	}
	__declspec(dllexport) int32_t StandardDeviationDouble(cuStats* stat, double &result, double *population, const int64_t population_size, double mean) {
		return marshal_cuda_error(stat->standard_deviation(result, population, population_size, mean));
	}

	__declspec(dllexport) int32_t SampleStandardDeviationFloat(cuStats* stat, double &result, float *sample, const int64_t sample_size, float mean) {
		return marshal_cuda_error(stat->sample_standard_deviation(result, sample, sample_size, mean));
	}
	__declspec(dllexport) int32_t SampleStandardDeviationDouble(cuStats* stat, double &result, double *sample, const int64_t sample_size, double mean) {
		return marshal_cuda_error(stat->sample_standard_deviation(result, sample, sample_size, mean));
	}

	__declspec(dllexport) int32_t CovarianceFloat(cuStats* stat, double &result, float *x_array, float x_mean, float *y_array, float y_mean, const int64_t array_size) {
		return marshal_cuda_error(stat->covariance(result, x_array, x_mean, y_array, y_mean, array_size));
	}
	__declspec(dllexport) int32_t CovarianceDouble(cuStats* stat, double &result, double *x_array, double x_mean, double *y_array, double y_mean, const int64_t array_size) {
		return marshal_cuda_error(stat->covariance(result, x_array, x_mean, y_array, y_mean, array_size));
	}

	__declspec(dllexport) int32_t SampleCovarianceFloat(cuStats* stat, double &result, float *x_array, float x_mean, float *y_array, float y_mean, const int64_t array_size) {
		return marshal_cuda_error(stat->sample_covariance(result, x_array, x_mean, y_array, y_mean, array_size));
	}
	__declspec(dllexport) int32_t SampleCovarianceDouble(cuStats* stat, double &result, double *x_array, double x_mean, double *y_array, double y_mean, const int64_t array_size) {
		return marshal_cuda_error(stat->sample_covariance(result, x_array, x_mean, y_array, y_mean, array_size));
	}

	__declspec(dllexport) int32_t PearsonCorrelationFloat(cuStats* stat, double &result, float *x_array, float x_mean, float *y_array, float y_mean, const int64_t array_size) {
		return marshal_cuda_error(stat->pearson_correlation(result, x_array, x_mean, y_array, y_mean, array_size));
	}
	__declspec(dllexport) int32_t PearsonCorrelationDouble(cuStats* stat, double &result, double *x_array, double x_mean, double *y_array, double y_mean, const int64_t array_size) {
		return marshal_cuda_error(stat->pearson_correlation(result, x_array, x_mean, y_array, y_mean, array_size));
	}
}
