#pragma once
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_profiler_api.h"
#include "device_functions.h"
#include "curand.h"
#include "curand_kernel.h"
#include <iostream>
#include <algorithm>
#include <time.h>
#include <chrono>

#include "DeviceInfo.h"
#include "AutomatedStreams.h"

class cuda_rand {
public:
	static cudaError_t uniform_rand(int32_t device_id, const int64_t amount_of_numbers, float *result);
	static cudaError_t uniform_rand_double(int32_t device_id, const int64_t amount_of_numbers, double *result);

	static cudaError_t normal_rand(int32_t device_id, const int64_t amount_of_numbers, float *result);
	static cudaError_t normal_rand_double(int32_t device_id, const int64_t amount_of_numbers, double *result);

	static cudaError_t log_normal_rand(int32_t device_id, const int64_t amount_of_numbers, float *result, float mean, float stddev);
	static cudaError_t log_normal_rand_double(int32_t device_id, const int64_t amount_of_numbers, double *result, double mean, double stddev);

	static cudaError_t poisson_rand(int32_t device_id, const int64_t amount_of_numbers, int32_t *result, double lambda);

protected:
	static void determine_launch_parameters(int32_t* blocks, int32_t* threads, const int64_t array_size, const int32_t max_block_size, const int32_t max_thread_size);
	static int64_t time_seed();
};

extern "C" {
	__declspec(dllexport) int32_t UniformRand(int32_t device_id, float *result, int64_t amount_of_numbers);
	__declspec(dllexport) int32_t UniformRandDouble(int32_t device_id, double *result, int64_t amount_of_numbers);
	__declspec(dllexport) int32_t NormalRand(int32_t device_id, float *result, int64_t amount_of_numbers);
	__declspec(dllexport) int32_t NormalRandDouble(int32_t device_id, double *result, int64_t amount_of_numbers);
	__declspec(dllexport) int32_t LogNormalRand(int32_t device_id, float *result, int64_t amount_of_numbers, float mean, float stddev);
	__declspec(dllexport) int32_t LogNormalRandDouble(int32_t device_id, double *result, int64_t amount_of_numbers, float mean, float stddev);
	__declspec(dllexport) int32_t PoissonRand(int32_t device_id, int32_t *result, int64_t amount_of_numbers, double lambda);
}

__global__ void cuda_rand_uniform_rand_kernel(int64_t seed, float *numbers, const int64_t maximum);
__global__ void cuda_rand_uniform_rand_double_kernel(int64_t seed, double *numbers, const int64_t maximum);
__global__ void cuda_rand_normal_rand_kernel(int64_t seed, float *numbers, const int64_t maximum);
__global__ void cuda_rand_normal_rand_double_kernel(int64_t seed, double *numbers, const int64_t maximum);
__global__ void cuda_rand_log_normal_rand_kernel(int64_t seed, float *numbers, const int64_t maximum, float mean, float stddev);
__global__ void cuda_rand_log_normal_rand_double_kernel(int64_t seed, double *numbers, const int64_t maximum, double mean, double stddev);
__global__ void cuda_rand_poisson_rand_kernel(int64_t seed, int32_t *numbers, const int64_t maximum, double lambda);