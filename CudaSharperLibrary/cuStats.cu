#pragma once
#include "cuStats.h"

namespace csl {

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

		T* h_result = (T*)malloc(data_size_in_memory);

		errorCode = cudaMemcpy(sample_ptr, sample, data_size_in_memory, cudaMemcpyHostToDevice);
		if (errorCode != cudaSuccess) return errorCode;

		kernels::cuStats::standard_deviation_kernel << <blocks, threads >> > (result_ptr, sample_ptr, sample_size, mean);

		errorCode = cudaMemcpy(h_result, result_ptr, data_size_in_memory, cudaMemcpyDeviceToHost);
		if (errorCode != cudaSuccess) return errorCode;

		double sum = 0;
		for (int64_t i = 0; i < sample_size; i++) {
			sum += h_result[i];
		}

		result = sum;

		//thrust::device_ptr<T> wrapped_ptr = thrust::device_pointer_cast(result_ptr);
		//result = (double)thrust::reduce(thrust::device, wrapped_ptr, wrapped_ptr + sample_size, (T)0, thrust::plus<T>());

		free(h_result);

		return cudaSuccess;
	}

	cudaError_t cuStats::standard_deviation(double& __restrict result, const float* __restrict sample, const int64_t sample_size, const float mean) {
		double sum = 0;
		cudaError_t errorCode = this->standard_deviation_summation(sum, sample, sample_size, mean);
		result = sqrt(sum / sample_size);
		return errorCode;
	}

	cudaError_t cuStats::standard_deviation(double& __restrict result, const double* __restrict sample, const int64_t sample_size, const double mean) {
		double sum = 0;
		cudaError_t errorCode = this->standard_deviation_summation(sum, sample, sample_size, mean);
		result = sqrt(sum / sample_size);
		return errorCode;
	}

	cudaError_t cuStats::sample_standard_deviation(double& __restrict result, const float* __restrict sample, const int64_t sample_size, const float mean) {
		double sum = 0;
		cudaError_t errorCode = this->standard_deviation_summation(sum, sample, sample_size, mean);
		result = sqrt(sum / (sample_size - 1));
		return errorCode;
	}

	cudaError_t cuStats::sample_standard_deviation(double& __restrict result, const double* __restrict sample, const int64_t sample_size, const double mean) {
		double sum = 0;
		cudaError_t errorCode = this->standard_deviation_summation(sum, sample, sample_size, mean);
		result = sqrt(sum / (sample_size - 1));
		return errorCode;
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

		kernels::cuStats::covariance_kernel << <blocks, threads, 0 >> > (result_ptr, x_ptr, x_mean, y_ptr, y_mean, array_size);

		thrust::device_ptr<T> wrapped_ptr = thrust::device_pointer_cast(result_ptr);
		result = (double)thrust::reduce(thrust::device, wrapped_ptr, wrapped_ptr + array_size, (T)0, thrust::plus<T>());

		return cudaSuccess;
	}

	cudaError_t cuStats::covariance_summation(
		double& __restrict result,
		const float* __restrict x_array, const float x_mean,
		const float* __restrict y_array, const float y_mean,
		const int64_t array_size) {
		return this->covariance_summation<float>(
			this->device_ptr_result->getf32(),
			this->device_ptr_x->getf32(),
			this->device_ptr_y->getf32(),
			result,
			x_array, x_mean,
			y_array, y_mean,
			array_size);
	}

	cudaError_t cuStats::covariance_summation(
		double& __restrict result,
		const double* __restrict x_array, const double x_mean,
		const double* __restrict y_array, const double y_mean,
		const int64_t array_size) {
		return this->covariance_summation<double>(
			this->device_ptr_result->getf64(),
			this->device_ptr_x->getf64(),
			this->device_ptr_y->getf64(),
			result,
			x_array, x_mean,
			y_array, y_mean,
			array_size);
	}

	cudaError_t cuStats::covariance(double& __restrict result, const float* __restrict x_array, const float x_mean, const float* __restrict y_array, const float y_mean, const int64_t array_size) {
		double sum = 0;
		cudaError_t errorCode = this->covariance_summation(sum, x_array, x_mean, y_array, y_mean, array_size);
		if (errorCode != cudaSuccess) return errorCode;
		result = sum / array_size;
		return errorCode;
	}

	cudaError_t cuStats::covariance(double& __restrict result, const double* __restrict x_array, const double x_mean, const double* __restrict y_array, const double y_mean, const int64_t array_size) {
		double sum = 0;
		cudaError_t errorCode = this->covariance_summation(sum, x_array, x_mean, y_array, y_mean, array_size);
		if (errorCode != cudaSuccess) return errorCode;
		result = sum / array_size;
		return errorCode;
	}

	cudaError_t cuStats::sample_covariance(double& __restrict result, const float* __restrict x_array, const float x_mean, const float* __restrict y_array, const float y_mean, const int64_t array_size) {
		double sum = 0;
		cudaError_t errorCode = this->covariance_summation(sum, x_array, x_mean, y_array, y_mean, array_size);
		if (errorCode != cudaSuccess) return errorCode;
		result = sum / (array_size - 1);
		return errorCode;
	}

	cudaError_t cuStats::sample_covariance(double& __restrict result, const double* __restrict x_array, const double x_mean, const double* __restrict y_array, const double y_mean, const int64_t array_size) {
		double sum = 0;
		cudaError_t errorCode = this->covariance_summation(sum, x_array, x_mean, y_array, y_mean, array_size);
		if (errorCode != cudaSuccess) return errorCode;
		result = sum / (array_size - 1);
		return errorCode;
	}

	cudaError_t cuStats::pearson_correlation(double& __restrict result, const float* __restrict x_array, const float x_mean, const float* __restrict y_array, const float y_mean, const int64_t array_size) {
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

	cudaError_t cuStats::pearson_correlation(double& __restrict result, const double* __restrict x_array, const double x_mean, const double* __restrict y_array, const double y_mean, const int64_t array_size) {
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

	namespace kernels {
		namespace cuStats {
			__global__ void standard_deviation_kernel(float* __restrict std, const float* __restrict sample, const int64_t data_set_size, const float mean) {
				int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

				for (int i = idx; i < data_set_size; i += blockDim.x * gridDim.x) {
					std[i] = (sample[i] - mean) * (sample[i] - mean);
				}

//				float4 std_data;
//
//#pragma unroll
//				for (int i = idx; i < data_set_size / 4; i += blockDim.x * gridDim.x) {
//					std_data.x = (sample[i] - mean) * (sample[i] - mean);
//					std_data.y = (sample[i + 1] - mean) * (sample[i + 1] - mean);
//					std_data.z = (sample[i + 2] - mean) * (sample[i + 2] - mean);
//					std_data.w = (sample[i + 3] - mean) * (sample[i + 3] - mean);
//					reinterpret_cast<float4*>(std)[i] = std_data;
//				}
//
//#pragma unroll
//				for (int i = idx + (data_set_size / 4) * 4; i < data_set_size; i += idx) {
//					std[i] = (sample[i] - mean) * (sample[i] - mean);
//				}
			}

			__global__ void standard_deviation_kernel(double* __restrict std, const double* __restrict sample, const int64_t data_set_size, const double mean) {
				int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
				for (int i = idx; i < data_set_size; i += blockDim.x * gridDim.x) {
					std[i] = (sample[i] - mean) * (sample[i] - mean);
				}
			}

			__global__ void covariance_kernel(float* __restrict result, const float* __restrict x_array, const float x_mean, const float* __restrict y_array, const float y_mean, const int64_t data_set_size) {
				int idx = threadIdx.x + (blockIdx.x * blockDim.x);
				for (int i = idx; i < data_set_size; i += blockDim.x * gridDim.x) {
					result[i] = (x_array[i] - x_mean) * (y_array[i] - y_mean);
				}
//				float4 cov_f32;
//
//#pragma unroll
//				for (int i = idx; i < data_set_size / 4; i += blockDim.x * gridDim.x) {
//					cov_f32.x = (x_array[i] - x_mean) * (y_array[i] - y_mean);
//					cov_f32.y = (x_array[i + 1] - x_mean) * (y_array[i + 1] - y_mean);
//					cov_f32.z = (x_array[i + 2] - x_mean) * (y_array[i + 2] - y_mean);
//					cov_f32.w = (x_array[i + 3] - x_mean) * (y_array[i + 3] - y_mean);
//					reinterpret_cast<float4*>(result)[i] = cov_f32;
//				}
//
//#pragma unroll
//				for (int i = idx + (data_set_size / 4) * 4; i < data_set_size; i += idx) {
//					result[i] = (x_array[i] - x_mean) * (y_array[i] - y_mean);
//				}
			}

			__global__ void covariance_kernel(double* __restrict result, const double* __restrict x_array, const double x_mean, const double* __restrict y_array, const double y_mean, const int64_t data_set_size) {
				int idx = threadIdx.x + (blockIdx.x * blockDim.x);
				for (int i = idx; i < data_set_size; i += blockDim.x * gridDim.x) {
					result[i] = (x_array[i] - x_mean) * (y_array[i] - y_mean);
				}
			}
		}
	}
}
