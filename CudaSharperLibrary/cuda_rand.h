#pragma once
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_profiler_api.h"
#include "device_functions.h"
#include "curand.h"
#include "curand_kernel.h"
#include <algorithm>
#include <time.h>
#include <chrono>

#include "cuda_mem_mgmt.h"
#include "launch_parameters.h"
#include "device_info.h"

// sizeof(curandState_t) = 48.
// Launching 64 threads, 48 * 64 = 3072; 3072 * 32 = 98304 bytes. (32 = active warps).
// 98304 is the amount of shared memory per block available on Pascal/Maxwell!
// Using the shared memory for this kernel can halve the execution time (on Pascal).
// For Kepler, we have to double the amount of threads to achieve 100% occupancy.
#define CURAND_NUM_OF_THREADS 64

namespace csl {
	// This class automatically allocates and frees device memory. This class is NOT thread-safe.
	class cuda_rand : cuda_launch_parameters {
	public:
		cuda_rand(int32_t device_id, int64_t amount_of_numbers) {
			this->device_id = device_id;
			this->device = std::make_unique<cuda_device>(device_id, amount_of_numbers);

			cudaDeviceProp prop;
			cudaGetDeviceProperties(&prop, device_id);

			// These kernels like blocks over threads.
			this->max_blocks = prop.multiProcessorCount * 32;
			this->max_threads = 64;

			if (prop.major == 3 && prop.minor == 5) {
				this->max_threads = 128;
			}
		}

		~cuda_rand() {
			this->device.reset();
		}

		int64_t max_size() const {
			return this->device->max_size();
		}

		cudaError_t uniform_rand(const int64_t amount_of_numbers, float* __restrict result);
		cudaError_t uniform_rand_double(const int64_t amount_of_numbers, double* __restrict result);

		cudaError_t normal_rand(const int64_t amount_of_numbers, float* __restrict result);
		cudaError_t normal_rand_double(const int64_t amount_of_numbers, double* __restrict result);

		cudaError_t log_normal_rand(const int64_t amount_of_numbers, float* __restrict result, const float mean, const float stddev);
		cudaError_t log_normal_rand_double(const int64_t amount_of_numbers, double* __restrict result, const double mean, const double stddev);

		cudaError_t poisson_rand(const int64_t amount_of_numbers, int32_t* __restrict result, const double lambda);

	protected:
		int32_t device_id;
		std::unique_ptr<cuda_device> device;

		void determine_launch_parameters(int32_t* blocks, int32_t* threads, const int64_t array_size, const int32_t max_block_size, const int32_t max_thread_size) {
			if ((*blocks) * (*threads) < array_size) {
				if ((*blocks * 2) < max_block_size) {
					*blocks = (*blocks * 2);
					this->determine_launch_parameters(blocks, threads, array_size, max_block_size, max_thread_size);
				}
				else if ((*threads * 2) < max_thread_size) {
					*threads = (*threads * 2);
					this->determine_launch_parameters(blocks, threads, array_size, max_block_size, max_thread_size);
				}
				return;
			}
			return;
		}

		int64_t time_seed() {
			// time(NULL) is not precise enough to produce different sets of random numbers.
			return std::chrono::system_clock::now().time_since_epoch().count();
		}
	};

	namespace kernels {
		namespace cuda_rand {
			__global__ void uniform_rand_kernel(const int64_t seed, float* __restrict numbers, const int64_t maximum);
			__global__ void uniform_rand_double_kernel(const int64_t seed, double* __restrict numbers, const int64_t maximum);
			__global__ void normal_rand_kernel(const int64_t seed, float* __restrict numbers, const int64_t maximum);
			__global__ void normal_rand_double_kernel(const int64_t seed, double* __restrict numbers, const int64_t maximum);
			__global__ void log_normal_rand_kernel(const int64_t seed, float* __restrict numbers, const int64_t maximum, const float mean, const float stddev);
			__global__ void log_normal_rand_double_kernel(const int64_t seed, double* __restrict numbers, const int64_t maximum, const double mean, const double stddev);
			__global__ void poisson_rand_kernel(const int64_t seed, int32_t* __restrict numbers, const int64_t maximum, const double lambda);
		}
	}
}
