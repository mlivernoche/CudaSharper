#pragma once
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_profiler_api.h"
#include "device_functions.h"
#include "thrust\reduce.h"
#include "thrust\execution_policy.h"
#include <atomic>
#include <math.h>

#include "cuda_mem_mgmt.h"
#include "launch_parameters.h"
#include "device_info.h"

#define CUSTATS_NUM_OF_THREADS 64
#define CUSTATS_MIN_NUM_OF_BLOCKS 2

namespace csl {
	class cuStats : cuda_launch_parameters {
	public:
		cuStats(int32_t device_id, int64_t amount_of_numbers) {
			this->device_id = device_id;
			this->device_ptr_x = std::make_unique<cuda_device>(device_id, amount_of_numbers);
			this->device_ptr_y = std::make_unique<cuda_device>(device_id, amount_of_numbers);
			this->device_ptr_result = std::make_unique<cuda_device>(device_id, amount_of_numbers);

			cudaDeviceProp prop;
			cudaGetDeviceProperties(&prop, device_id);

			// These kernels like blocks over threads.
			this->max_blocks = prop.multiProcessorCount * 32;
			this->max_threads = prop.maxThreadsPerBlock;
		}

		~cuStats() {
			this->device_ptr_x.reset();
			this->device_ptr_y.reset();
			this->device_ptr_result.reset();
		}

		int64_t max_size() const {
			return this->device_ptr_result->max_size();
		}

		// The only difference between pop STD and sample STD is the n - 1.
		cudaError_t standard_deviation(double& __restrict result, const float* __restrict sample, const int64_t sample_size, const float mean);
		cudaError_t standard_deviation(double& __restrict result, const double* __restrict sample, const int64_t sample_size, double mean);

		cudaError_t sample_standard_deviation(double& __restrict result, const float* __restrict sample, const int64_t sample_size, const float mean);
		cudaError_t sample_standard_deviation(double& __restrict result, const double* __restrict sample, const int64_t sample_size, const double mean);

		cudaError_t covariance(double& __restrict result, const float* __restrict x_array, const float x_mean, const float* __restrict y_array, const float y_mean, const int64_t array_size);
		cudaError_t covariance(double& __restrict result, const double* __restrict x_array, const double x_mean, const double* __restrict y_array, const double y_mean, const int64_t array_size);

		cudaError_t sample_covariance(double& __restrict result, const float* __restrict x_array, const float x_mean, const float* __restrict y_array, const float y_mean, const int64_t array_size);
		cudaError_t sample_covariance(double& __restrict result, const double* __restrict x_array, const double x_mean, const double* __restrict y_array, const double y_mean, const int64_t array_size);

		cudaError_t pearson_correlation(double& __restrict result, const float* __restrict x_array, const float x_mean, const float* __restrict y_array, const float y_mean, const int64_t array_size);
		cudaError_t pearson_correlation(double& __restrict result, const double* __restrict x_array, const double x_mean, const double* __restrict y_array, const double y_mean, const int64_t array_size);
	protected:
		int32_t device_id;
		std::unique_ptr<cuda_device> device_ptr_x;
		std::unique_ptr<cuda_device> device_ptr_y;
		std::unique_ptr<cuda_device> device_ptr_result;

		void determine_launch_parameters(int32_t* blocks, int32_t* threads, const int64_t array_size, const int32_t max_block_size, const int32_t max_thread_size) {
			if (*blocks * *threads < array_size) {
				if ((*threads * 2) <= max_thread_size)
				{
					*threads = (*threads * 2);
					this->determine_launch_parameters(blocks, threads, array_size, max_block_size, max_thread_size);
				}
				else if ((*blocks * 2) <= max_block_size)
				{
					*blocks = (*blocks * 2);
					this->determine_launch_parameters(blocks, threads, array_size, max_block_size, max_thread_size);
				}
				return;
			}
			return;
		}

	private:
		template<typename T>
		cudaError_t standard_deviation_summation(
			T* __restrict result_ptr,
			T* __restrict sample_ptr,
			double& __restrict result,
			const T* __restrict sample, const int64_t sample_size,
			const T mean);
		cudaError_t standard_deviation_summation(
			double& __restrict result,
			const float* __restrict sample, const int64_t sample_size,
			const float mean) {
			return this->standard_deviation_summation(
				this->device_ptr_result->getf32(),
				this->device_ptr_y->getf32(),
				result,
				sample, sample_size,
				mean);
		}
		cudaError_t standard_deviation_summation(
			double& __restrict result,
			const double* __restrict sample, const int64_t sample_size,
			const double mean) {
			return this->standard_deviation_summation(
				this->device_ptr_result->getf64(),
				this->device_ptr_y->getf64(),
				result,
				sample, sample_size,
				mean);
		}

		template<typename T>
		cudaError_t covariance_summation(
			T* __restrict result_ptr,
			T* __restrict x_ptr,
			T* __restrict y_ptr,
			double& __restrict result,
			const T* __restrict x_array, const T x_mean,
			const T* __restrict y_array, const T y_mean,
			const int64_t array_size);
		cudaError_t covariance_summation(
			double& __restrict result,
			const float* __restrict x_array, const float x_mean,
			const float* __restrict y_array, const float y_mean,
			const int64_t array_size);
		cudaError_t covariance_summation(
			double& __restrict result,
			const double* __restrict x_array, const double x_mean,
			const double* __restrict y_array, const double y_mean,
			const int64_t array_size);
	};

	namespace kernels {
		namespace cuStats {
			__global__ void standard_deviation_kernel(float* __restrict std, const float* __restrict sample, const int64_t data_set_size, const float mean);
			__global__ void standard_deviation_kernel(double* __restrict std, const double* __restrict sample, const int64_t data_set_size, const double mean);

			__global__ void covariance_kernel(double* __restrict result, const double* __restrict x_array, const double x_mean, const double* __restrict y_array, const double y_mean, const int64_t data_set_size);
			__global__ void covariance_kernel(float* __restrict result, const float* __restrict x_array, const float x_mean, const float* __restrict y_array, const float y_mean, const int64_t data_set_size);
		}
	}
}

