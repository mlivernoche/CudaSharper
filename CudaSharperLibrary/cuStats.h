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
#include "DeviceInfo.h"

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

	// The only difference between pop STD and sample STD is the n - 1.
	template<typename T>
	cudaError_t standard_deviation(double& __restrict result, const T* __restrict sample, const int64_t sample_size, const T mean);

	template<typename T>
	cudaError_t sample_standard_deviation(double& __restrict result, const T* __restrict sample, const int64_t sample_size, const T mean);

	template<typename T>
	cudaError_t covariance(double& __restrict result, const T* __restrict x_array, const T x_mean, const T* __restrict y_array, const T y_mean, const int64_t array_size);
	template<typename T>
	cudaError_t sample_covariance(double& __restrict result, const T* __restrict x_array, const T x_mean, const T* __restrict y_array, const T y_mean, const int64_t array_size);

	template<typename T>
	cudaError_t pearson_correlation(double& __restrict result, const T* __restrict x_array, const T x_mean, const T* __restrict y_array, const T y_mean, const int64_t array_size);
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
		const int64_t array_size) {
		return this->covariance_summation(
			this->device_ptr_result->getf32(),
			this->device_ptr_x->getf32(),
			this->device_ptr_y->getf32(),
			result,
			x_array, x_mean,
			y_array, y_mean,
			array_size);
	}
	cudaError_t covariance_summation(
		double& __restrict result,
		const double* __restrict x_array, const double x_mean,
		const double* __restrict y_array, const double y_mean,
		const int64_t array_size) {
		return this->covariance_summation(
			this->device_ptr_result->getf64(),
			this->device_ptr_x->getf64(),
			this->device_ptr_y->getf64(),
			result,
			x_array, x_mean,
			y_array, y_mean,
			array_size);
	}
};

__global__ void cuStats_standard_deviation_kernel(float* __restrict std, const float* __restrict sample, const int64_t data_set_size, const float mean);
__global__ void cuStats_standard_deviation_kernel(double* __restrict std, const double* __restrict sample, const int64_t data_set_size, const double mean);

__global__ void cuStats_covariance_kernel(double* __restrict result, const double* __restrict x_array, const double x_mean, const double* __restrict y_array, const double y_mean, const int64_t data_set_size);
__global__ void cuStats_covariance_kernel(float* __restrict result, const float* __restrict x_array, const float x_mean, const float* __restrict y_array, const float y_mean, const int64_t data_set_size);

// C does not "support" function overloading like C++ does.
// Why, then, do these have to be marked as C? C++ will mangle the function names to support overloading.
// Marking them as C will make sure that these function names will not be changed.
extern "C" {
	__declspec(dllexport) cuStats* CreateStatClass(int32_t device_id, int64_t amount_of_numbers);
	__declspec(dllexport) void DisposeStatClass(cuStats* stat);

	__declspec(dllexport) int32_t StandardDeviationFloat(cuStats* stat, double &result, float *population, const int64_t population_size, float mean);
	__declspec(dllexport) int32_t StandardDeviationDouble(cuStats* stat, double &result, double *population, const int64_t population_size, double mean);

	__declspec(dllexport) int32_t SampleStandardDeviationFloat(cuStats* stat, double &result, float *sample, const int64_t sample_size, float mean);
	__declspec(dllexport) int32_t SampleStandardDeviationDouble(cuStats* stat, double &result, double *sample, const int64_t sample_size, double mean);

	__declspec(dllexport) int32_t SampleCovarianceFloat(cuStats* stat, double &result, float *x_array, float x_mean, float *y_array, float y_mean, const int64_t array_size);
	__declspec(dllexport) int32_t SampleCovarianceDouble(cuStats* stat, double &result, double *x_array, double x_mean, double *y_array, double y_mean, const int64_t array_size);

	__declspec(dllexport) int32_t CovarianceFloat(cuStats* stat, double &result, float *x_array, float x_mean, float *y_array, float y_mean, const int64_t array_size);
	__declspec(dllexport) int32_t CovarianceDouble(cuStats* stat, double &result, double *x_array, double x_mean, double *y_array, double y_mean, const int64_t array_size);

	__declspec(dllexport) int32_t PearsonCorrelationFloat(cuStats* stat, double &result, float *x_array, float x_mean, float *y_array, float y_mean, const int64_t array_size);
	__declspec(dllexport) int32_t PearsonCorrelationDouble(cuStats* stat, double &result, double *x_array, double x_mean, double *y_array, double y_mean, const int64_t array_size);
}
