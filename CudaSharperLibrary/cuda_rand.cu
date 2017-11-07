#pragma once
#include "cuda_rand.h"

namespace csl {
	
	cudaError_t cuda_rand::uniform_rand(const int64_t amount_of_numbers, float* __restrict result) {
		cudaError_t errorCode = cudaSetDevice(this->device->device_id());
		if (errorCode != cudaSuccess) return errorCode;

		// kernel prefers blocks over threads, but does not like only blocks and no threads.
		int32_t threads = CURAND_NUM_OF_THREADS;
		int32_t blocks = 2;
		this->determine_launch_parameters(&blocks, &threads, amount_of_numbers, this->max_blocks, this->max_threads);

		size_t sharedMem = sizeof(curandState_t) * threads;
		kernels::cuda_rand::uniform_rand_kernel << <blocks, threads, sharedMem >> > (cuda_rand::time_seed(), this->device->getf32(), amount_of_numbers);

		errorCode = cudaMemcpy(result, this->device->getf32(), amount_of_numbers * sizeof(float), cudaMemcpyDeviceToHost);
		if (errorCode != cudaSuccess) return errorCode;

		return cudaSuccess;
	}

	cudaError_t cuda_rand::uniform_rand_double(const int64_t amount_of_numbers, double* __restrict result) {
		cudaError_t errorCode = cudaSetDevice(this->device->device_id());
		if (errorCode != cudaSuccess) return errorCode;

		// kernel prefers blocks over threads, but does not like only blocks and no threads.
		int32_t threads = CURAND_NUM_OF_THREADS;
		int32_t blocks = 2;
		this->determine_launch_parameters(&blocks, &threads, amount_of_numbers, this->max_blocks, this->max_threads);

		size_t sharedMem = sizeof(curandState_t) * threads;
		kernels::cuda_rand::uniform_rand_double_kernel << <blocks, threads, sharedMem >> > (cuda_rand::time_seed(), this->device->getf64(), amount_of_numbers);

		errorCode = cudaMemcpy(result, this->device->getf64(), amount_of_numbers * sizeof(double), cudaMemcpyDeviceToHost);
		if (errorCode != cudaSuccess) return errorCode;

		return cudaSuccess;
	}

	cudaError_t cuda_rand::normal_rand(const int64_t amount_of_numbers, float* __restrict result) {
		cudaError_t errorCode = cudaSetDevice(this->device->device_id());
		if (errorCode != cudaSuccess) return errorCode;

		int32_t threads = CURAND_NUM_OF_THREADS;
		int32_t blocks = 2;
		this->determine_launch_parameters(&blocks, &threads, amount_of_numbers, this->max_blocks, this->max_threads);

		size_t sharedMem = sizeof(curandState_t) * threads;
		kernels::cuda_rand::normal_rand_kernel << <blocks, threads, sharedMem >> > (cuda_rand::time_seed(), this->device->getf32(), amount_of_numbers);

		errorCode = cudaMemcpy(result, this->device->getf32(), sizeof(float) * amount_of_numbers, cudaMemcpyDeviceToHost);
		if (errorCode != cudaSuccess) return errorCode;

		return cudaSuccess;
	}

	cudaError_t cuda_rand::normal_rand_double(const int64_t amount_of_numbers, double* __restrict result) {
		cudaError_t errorCode = cudaSetDevice(this->device->device_id());
		if (errorCode != cudaSuccess) return errorCode;

		int32_t threads = CURAND_NUM_OF_THREADS;
		int32_t blocks = 2;
		this->determine_launch_parameters(&blocks, &threads, amount_of_numbers, this->max_blocks, this->max_threads);

		size_t sharedMem = sizeof(curandState_t) * threads;
		kernels::cuda_rand::normal_rand_double_kernel << <blocks, threads, sharedMem >> > (cuda_rand::time_seed(), this->device->getf64(), amount_of_numbers);

		errorCode = cudaMemcpy(result, this->device->getf64(), sizeof(double) * amount_of_numbers, cudaMemcpyDeviceToHost);
		if (errorCode != cudaSuccess) return errorCode;

		return cudaSuccess;
	}

	cudaError_t cuda_rand::log_normal_rand(const int64_t amount_of_numbers, float* __restrict result, float mean, float stddev) {
		cudaError_t errorCode = cudaSetDevice(this->device->device_id());
		if (errorCode != cudaSuccess) return errorCode;

		int32_t threads = CURAND_NUM_OF_THREADS;
		int32_t blocks = 2;
		this->determine_launch_parameters(&blocks, &threads, amount_of_numbers, this->max_blocks, this->max_threads);

		size_t sharedMem = sizeof(curandState_t) * threads;
		kernels::cuda_rand::log_normal_rand_kernel << <blocks, threads, sharedMem >> > (cuda_rand::time_seed(), this->device->getf32(), amount_of_numbers, mean, stddev);

		errorCode = cudaMemcpy(result, this->device->getf32(), sizeof(float) * amount_of_numbers, cudaMemcpyDeviceToHost);
		if (errorCode != cudaSuccess) return errorCode;

		return cudaSuccess;
	}

	cudaError_t cuda_rand::log_normal_rand_double(const int64_t amount_of_numbers, double* __restrict result, double mean, double stddev) {
		cudaError_t errorCode = cudaSetDevice(this->device->device_id());
		if (errorCode != cudaSuccess) return errorCode;

		int32_t threads = CURAND_NUM_OF_THREADS;
		int32_t blocks = 2;
		this->determine_launch_parameters(&blocks, &threads, amount_of_numbers, this->max_blocks, this->max_threads);

		size_t sharedMem = sizeof(curandState_t) * threads;
		kernels::cuda_rand::log_normal_rand_double_kernel << <blocks, threads, sharedMem >> > (cuda_rand::time_seed(), this->device->getf64(), amount_of_numbers, mean, stddev);

		errorCode = cudaMemcpy(result, this->device->getf64(), sizeof(double) * amount_of_numbers, cudaMemcpyDeviceToHost);
		if (errorCode != cudaSuccess) return errorCode;

		return cudaSuccess;
	}

	cudaError_t cuda_rand::poisson_rand(const int64_t amount_of_numbers, int32_t* __restrict result, double lambda) {
		cudaError_t errorCode = cudaSetDevice(this->device->device_id());
		if (errorCode != cudaSuccess) return errorCode;

		int32_t threads = CURAND_NUM_OF_THREADS;
		int32_t blocks = 2;
		this->determine_launch_parameters(&blocks, &threads, amount_of_numbers, this->max_blocks, this->max_threads);

		size_t sharedMem = sizeof(curandState_t) * threads;
		kernels::cuda_rand::poisson_rand_kernel << <blocks, threads, sharedMem >> > (cuda_rand::time_seed(), this->device->getu32(), amount_of_numbers, lambda);

		errorCode = cudaMemcpy(result, this->device->getu32(), sizeof(float) * amount_of_numbers, cudaMemcpyDeviceToHost);
		if (errorCode != cudaSuccess) return errorCode;

		return cudaSuccess;
	}

	namespace kernels {
		namespace cuda_rand {
			__global__ void uniform_rand_kernel(const int64_t seed, float* __restrict numbers, const int64_t maximum) {
				extern __shared__ curandState_t curandStateShared[];

				int xid = blockIdx.x * blockDim.x + threadIdx.x;
				curand_init(seed + xid, 0, 0, &curandStateShared[threadIdx.x]);

				for (int i = xid; i < maximum; i += blockDim.x * gridDim.x) {
					numbers[i] = curand_uniform(&curandStateShared[threadIdx.x]);
				}
			}

			__global__ void uniform_rand_double_kernel(const int64_t seed, double* __restrict numbers, const int64_t maximum) {
				extern __shared__ curandState_t curandStateShared[];

				int xid = blockIdx.x * blockDim.x + threadIdx.x;
				curand_init(seed + xid, 0, 0, &curandStateShared[threadIdx.x]);

				for (int i = xid; i < maximum; i += blockDim.x * gridDim.x) {
					numbers[i] = curand_uniform_double(&curandStateShared[threadIdx.x]);
				}
			}

			__global__ void normal_rand_kernel(const int64_t seed, float* __restrict numbers, const int64_t maximum) {
				extern __shared__ curandState_t curandStateShared[];

				int idx = blockIdx.x * blockDim.x + threadIdx.x;
				int N = maximum / 2;
				curand_init(seed + idx, 0, 0, &curandStateShared[threadIdx.x]);

				for (int i = idx; i < N; i += blockDim.x * gridDim.x) {
					// We could use curand_normal4, but it requires curandStatePhilox4_32_10_t
					// That struct is 64 bytes. Using Pascal, that will lead to a maximum theoretical occupancy of 75%.
					// Using curandState_t, which is 48 bytes, we can achieve a occupancy of 100%.
					// This kernel is compute-bound, so achieving higher memory bandwidth over compute will not improve performance.
					reinterpret_cast<float2*>(numbers)[i] = curand_normal2(&curandStateShared[threadIdx.x]);
				}

				for (int i = idx + N * 2; i < maximum; i += idx) {
					numbers[i] = curand_normal(&curandStateShared[threadIdx.x]);
				}
			}

			__global__ void normal_rand_double_kernel(const int64_t seed, double* __restrict numbers, const int64_t maximum) {
				extern __shared__ curandState_t curandStateShared[];

				int idx = blockIdx.x * blockDim.x + threadIdx.x;
				int N = maximum / 2;
				curand_init(seed + idx, 0, 0, &curandStateShared[threadIdx.x]);

				for (int i = idx; i < N; i += blockDim.x * gridDim.x) {
					reinterpret_cast<double2*>(numbers)[i] = curand_normal2_double(&curandStateShared[threadIdx.x]);
				}

				for (int i = idx + N * 2; i < maximum; i += idx) {
					numbers[i] = curand_normal_double(&curandStateShared[threadIdx.x]);
				}
			}

			__global__ void log_normal_rand_kernel(const int64_t seed, float* __restrict numbers, const int64_t maximum, const float mean, const float stddev) {
				extern __shared__ curandState_t curandStateShared[];

				int idx = blockIdx.x * blockDim.x + threadIdx.x;
				int N = maximum / 2;
				curand_init(seed + idx, 0, 0, &curandStateShared[threadIdx.x]);

				for (int i = idx; i < N; i += blockDim.x * gridDim.x) {
					reinterpret_cast<float2*>(numbers)[i] = curand_log_normal2(&curandStateShared[threadIdx.x], mean, stddev);
				}

				for (int i = idx + N * 2; i < maximum; i += idx) {
					numbers[i] = curand_log_normal(&curandStateShared[threadIdx.x], mean, stddev);
				}
			}

			__global__ void log_normal_rand_double_kernel(const int64_t seed, double* __restrict numbers, const int64_t maximum, const double mean, const double stddev) {
				extern __shared__ curandState_t curandStateShared[];

				int idx = blockIdx.x * blockDim.x + threadIdx.x;
				int N = maximum / 2;
				curand_init(seed + idx, 0, 0, &curandStateShared[threadIdx.x]);

				for (int i = idx; i < N; i += blockDim.x * gridDim.x) {
					reinterpret_cast<double2*>(numbers)[i] = curand_log_normal2_double(&curandStateShared[threadIdx.x], mean, stddev);
				}

				for (int i = idx + N * 2; i < maximum; i += idx) {
					numbers[i] = curand_log_normal_double(&curandStateShared[threadIdx.x], mean, stddev);
				}
			}

			__global__ void poisson_rand_kernel(const int64_t seed, int32_t* __restrict numbers, const int64_t maximum, const double lambda) {
				extern __shared__ curandState_t curandStateShared[];

				int xid = blockIdx.x * blockDim.x + threadIdx.x;
				curand_init(seed + xid, 0, 0, &curandStateShared[threadIdx.x]);

				for (int i = xid; i < maximum; i += blockDim.x * gridDim.x) {
					numbers[i] = curand_poisson(&curandStateShared[threadIdx.x], lambda);
				}
			}
		}
	}
}
