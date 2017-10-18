#pragma once
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_profiler_api.h"
#include "device_functions.h"
#include <atomic>
#include <math.h>

#include "DeviceInfo.h"

class cuStats {
protected:
	static void determine_launch_parameters(int32_t* __restrict blocks, int32_t* __restrict threads, const int64_t array_size, const int32_t max_block_size, const int32_t max_thread_size);
	
	template<typename T> static cudaError_t standard_deviation_summation(const int32_t device_id, double& __restrict result, const T* __restrict sample, const int64_t sample_size, const T mean);
	template<typename T> static cudaError_t covariance_summation(const int32_t device_id, double& __restrict result, const T* __restrict x_array, const T x_mean, const T* __restrict y_array, const T y_mean, const int64_t array_size);

public:
	// The only difference between pop STD and sample STD is the n - 1.
	template<typename T> static cudaError_t standard_deviation(const int32_t device_id, double& __restrict result, const T* __restrict sample, const int64_t sample_size, const T mean);
	template<typename T> static cudaError_t sample_standard_deviation(const int32_t device_id, double& __restrict result, const T* __restrict sample, const int64_t sample_size, const T mean);

	template<typename T> static cudaError_t covariance(const int32_t device_id, double& __restrict result, const T* __restrict x_array, const T x_mean, const T* __restrict y_array, const T y_mean, const int64_t array_size);
	template<typename T> static cudaError_t sample_covariance(const int32_t device_id, double& __restrict result, const T* __restrict x_array, const T x_mean, const T* __restrict y_array, const T y_mean, const int64_t array_size);

	template<typename T> static cudaError_t pearson_correlation(const int32_t device_id, double& __restrict result, const T* __restrict x_array, const T x_mean, const T* __restrict y_array, const T y_mean, const int64_t array_size);
};

__global__ void cuStats_standard_deviation_kernel(float* __restrict std, const float* __restrict sample, const int64_t data_set_size, const float mean);
__global__ void cuStats_standard_deviation_kernel(double* __restrict std, const double* __restrict sample, const int64_t data_set_size, const double mean);

__global__ void cuStats_covariance_kernel(double* __restrict result, const double* __restrict x_array, const double x_mean, const double* __restrict y_array, const double y_mean, const int64_t data_set_size);
__global__ void cuStats_covariance_kernel(float* __restrict result, const float* __restrict x_array, const float x_mean, const float* __restrict y_array, const float y_mean, const int64_t data_set_size);

// C does not "support" function overloading like C++ does.
// Why, then, do these have to be marked as C? C++ will mangle the function names to support overloading.
// Marking them as C will make sure that these function names will not be changed.
extern "C" {
	__declspec(dllexport) int32_t StandardDeviationFloat(int32_t device_id, double &result, float *population, const int64_t population_size, float mean);
	__declspec(dllexport) int32_t StandardDeviationDouble(int32_t device_id, double &result, double *population, const int64_t population_size, double mean);

	__declspec(dllexport) int32_t SampleStandardDeviationFloat(int32_t device_id, double &result, float *sample, const int64_t sample_size, float mean);
	__declspec(dllexport) int32_t SampleStandardDeviationDouble(int32_t device_id, double &result, double *sample, const int64_t sample_size, double mean);

	__declspec(dllexport) int32_t SampleCovarianceFloat(int32_t device_id, double &result, float *x_array, float x_mean, float *y_array, float y_mean, const int64_t array_size);
	__declspec(dllexport) int32_t SampleCovarianceDouble(int32_t device_id, double &result, double *x_array, double x_mean, double *y_array, double y_mean, const int64_t array_size);

	__declspec(dllexport) int32_t CovarianceFloat(int32_t device_id, double &result, float *x_array, float x_mean, float *y_array, float y_mean, const int64_t array_size);
	__declspec(dllexport) int32_t CovarianceDouble(int32_t device_id, double &result, double *x_array, double x_mean, double *y_array, double y_mean, const int64_t array_size);

	__declspec(dllexport) int32_t PearsonCorrelationFloat(int32_t device_id, double &result, float *x_array, float x_mean, float *y_array, float y_mean, const int64_t array_size);
	__declspec(dllexport) int32_t PearsonCorrelationDouble(int32_t device_id, double &result, double *x_array, double x_mean, double *y_array, double y_mean, const int64_t array_size);
}
