#pragma once
#include "cuda_rand.h"

// sizeof(curandState_t) = 48.
// Launching 64 threads, 48 * 64 = 3072; 3072 * 32 = 98304 bytes. (32 = active warps).
// 98304 is the amount of shared memory per block available on Pascal/Maxwell!
// Using the shared memory for this kernel can halve the execution time (on Pascal).
// For Kepler, we have to double the amount of threads to achieve 100% occupancy.
#define CURAND_NUM_OF_THREADS 64

__global__ void cuda_rand_uniform_rand_kernel(const int64_t seed, float* __restrict numbers, const int64_t maximum) {
	extern __shared__ curandState_t curandStateShared[];

	int xid = blockIdx.x * blockDim.x + threadIdx.x;
	curand_init(seed + xid, 0, 0, &curandStateShared[threadIdx.x]);

	for (int i = xid; i < maximum; i += blockDim.x * gridDim.x) {
		numbers[i] = curand_uniform(&curandStateShared[threadIdx.x]);
	}
}

cudaError_t cuda_rand::uniform_rand(const int64_t amount_of_numbers, float* __restrict result) {
	cudaError_t errorCode = cudaSetDevice(this->device->device_id());
	if (errorCode != cudaSuccess) return errorCode;

	// kernel prefers blocks over threads, but does not like only blocks and no threads.
	int32_t threads = CURAND_NUM_OF_THREADS;
	int32_t blocks = 2;
	this->determine_launch_parameters(&blocks, &threads, amount_of_numbers, this->max_blocks, this->max_threads);

	size_t sharedMem = sizeof(curandState_t) * threads;
	cuda_rand_uniform_rand_kernel << <blocks, threads, sharedMem >> > (cuda_rand::time_seed(), this->device->getf32(), amount_of_numbers);

	errorCode = cudaMemcpy(result, this->device->getf32(), amount_of_numbers * sizeof(float), cudaMemcpyDeviceToHost);
	if (errorCode != cudaSuccess) return errorCode;

	return cudaSuccess;
}

__global__ void cuda_rand_uniform_rand_double_kernel(const int64_t seed, double* __restrict numbers, const int64_t maximum) {
	extern __shared__ curandState_t curandStateShared[];

	int xid = blockIdx.x * blockDim.x + threadIdx.x;
	curand_init(seed + xid, 0, 0, &curandStateShared[threadIdx.x]);

	for (int i = xid; i < maximum; i += blockDim.x * gridDim.x) {
		numbers[i] = curand_uniform_double(&curandStateShared[threadIdx.x]);
	}
}

cudaError_t cuda_rand::uniform_rand_double(const int64_t amount_of_numbers, double* __restrict result) {
	cudaError_t errorCode = cudaSetDevice(this->device->device_id());
	if (errorCode != cudaSuccess) return errorCode;

	// kernel prefers blocks over threads, but does not like only blocks and no threads.
	int32_t threads = CURAND_NUM_OF_THREADS;
	int32_t blocks = 2;
	this->determine_launch_parameters(&blocks, &threads, amount_of_numbers, this->max_blocks, this->max_threads);

	size_t sharedMem = sizeof(curandState_t) * threads;
	cuda_rand_uniform_rand_double_kernel << <blocks, threads, sharedMem >> > (cuda_rand::time_seed(), this->device->getf64(), amount_of_numbers);

	errorCode = cudaMemcpy(result, this->device->getf64(), amount_of_numbers * sizeof(double), cudaMemcpyDeviceToHost);
	if (errorCode != cudaSuccess) return errorCode;

	return cudaSuccess;
}

__global__ void cuda_rand_normal_rand_kernel(const int64_t seed, float* __restrict numbers, const int64_t maximum) {
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

cudaError_t cuda_rand::normal_rand(const int64_t amount_of_numbers, float* __restrict result) {
	cudaError_t errorCode = cudaSetDevice(this->device->device_id());
	if (errorCode != cudaSuccess) return errorCode;

	int32_t threads = CURAND_NUM_OF_THREADS;
	int32_t blocks = 2;
	this->determine_launch_parameters(&blocks, &threads, amount_of_numbers, this->max_blocks, this->max_threads);

	size_t sharedMem = sizeof(curandState_t) * threads;
	cuda_rand_normal_rand_kernel << <blocks, threads, sharedMem >> > (cuda_rand::time_seed(), this->device->getf32(), amount_of_numbers);

	errorCode = cudaMemcpy(result, this->device->getf32(), sizeof(float) * amount_of_numbers, cudaMemcpyDeviceToHost);
	if (errorCode != cudaSuccess) return errorCode;

	return cudaSuccess;
}

__global__ void cuda_rand_normal_rand_double_kernel(const int64_t seed, double* __restrict numbers, const int64_t maximum) {
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

cudaError_t cuda_rand::normal_rand_double(const int64_t amount_of_numbers, double* __restrict result) {
	cudaError_t errorCode = cudaSetDevice(this->device->device_id());
	if (errorCode != cudaSuccess) return errorCode;

	int32_t threads = CURAND_NUM_OF_THREADS;
	int32_t blocks = 2;
	this->determine_launch_parameters(&blocks, &threads, amount_of_numbers, this->max_blocks, this->max_threads);

	size_t sharedMem = sizeof(curandState_t) * threads;
	cuda_rand_normal_rand_double_kernel << <blocks, threads, sharedMem >> > (cuda_rand::time_seed(), this->device->getf64(), amount_of_numbers);

	errorCode = cudaMemcpy(result, this->device->getf64(), sizeof(double) * amount_of_numbers, cudaMemcpyDeviceToHost);
	if (errorCode != cudaSuccess) return errorCode;

	return cudaSuccess;
}

__global__ void cuda_rand_log_normal_rand_kernel(const int64_t seed, float* __restrict numbers, const int64_t maximum, const float mean, const float stddev) {
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

cudaError_t cuda_rand::log_normal_rand(const int64_t amount_of_numbers, float* __restrict result, float mean, float stddev) {
	cudaError_t errorCode = cudaSetDevice(this->device->device_id());
	if (errorCode != cudaSuccess) return errorCode;

	int32_t threads = CURAND_NUM_OF_THREADS;
	int32_t blocks = 2;
	this->determine_launch_parameters(&blocks, &threads, amount_of_numbers, this->max_blocks, this->max_threads);

	size_t sharedMem = sizeof(curandState_t) * threads;
	cuda_rand_log_normal_rand_kernel << <blocks, threads, sharedMem >> > (cuda_rand::time_seed(), this->device->getf32(), amount_of_numbers, mean, stddev);

	errorCode = cudaMemcpy(result, this->device->getf32(), sizeof(float) * amount_of_numbers, cudaMemcpyDeviceToHost);
	if (errorCode != cudaSuccess) return errorCode;

	return cudaSuccess;
}

__global__ void cuda_rand_log_normal_rand_double_kernel(const int64_t seed, double* __restrict numbers, const int64_t maximum, const double mean, const double stddev) {
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

cudaError_t cuda_rand::log_normal_rand_double(const int64_t amount_of_numbers, double* __restrict result, double mean, double stddev) {
	cudaError_t errorCode = cudaSetDevice(this->device->device_id());
	if (errorCode != cudaSuccess) return errorCode;

	int32_t threads = CURAND_NUM_OF_THREADS;
	int32_t blocks = 2;
	this->determine_launch_parameters(&blocks, &threads, amount_of_numbers, this->max_blocks, this->max_threads);

	size_t sharedMem = sizeof(curandState_t) * threads;
	cuda_rand_log_normal_rand_double_kernel << <blocks, threads, sharedMem >> > (cuda_rand::time_seed(), this->device->getf64(), amount_of_numbers, mean, stddev);

	errorCode = cudaMemcpy(result, this->device->getf64(), sizeof(double) * amount_of_numbers, cudaMemcpyDeviceToHost);
	if (errorCode != cudaSuccess) return errorCode;

	return cudaSuccess;
}

__global__ void cuda_rand_poisson_rand_kernel(const int64_t seed, int32_t* __restrict numbers, const int64_t maximum, const double lambda) {
	extern __shared__ curandState_t curandStateShared[];

	int xid = blockIdx.x * blockDim.x + threadIdx.x;
	curand_init(seed + xid, 0, 0, &curandStateShared[threadIdx.x]);

	for (int i = xid; i < maximum; i += blockDim.x * gridDim.x) {
		numbers[i] = curand_poisson(&curandStateShared[threadIdx.x], lambda);
	}
}

cudaError_t cuda_rand::poisson_rand(const int64_t amount_of_numbers, int32_t* __restrict result, double lambda) {
	cudaError_t errorCode = cudaSetDevice(this->device->device_id());
	if (errorCode != cudaSuccess) return errorCode;

	int32_t threads = CURAND_NUM_OF_THREADS;
	int32_t blocks = 2;
	this->determine_launch_parameters(&blocks, &threads, amount_of_numbers, this->max_blocks, this->max_threads);

	size_t sharedMem = sizeof(curandState_t) * threads;
	cuda_rand_poisson_rand_kernel << <blocks, threads, sharedMem >> > (cuda_rand::time_seed(), this->device->getu32(), amount_of_numbers, lambda);

	errorCode = cudaMemcpy(result, this->device->getu32(), sizeof(float) * amount_of_numbers, cudaMemcpyDeviceToHost);
	if (errorCode != cudaSuccess) return errorCode;

	return cudaSuccess;
}

extern "C" {
	__declspec(dllexport) cuda_rand* CreateRandomClass(int32_t device_id, int64_t amount_of_numbers) {
		return new cuda_rand(device_id, amount_of_numbers);
	}
	__declspec(dllexport) void DisposeRandomClass(cuda_rand* rand) {
		if (rand != NULL) {
			delete rand;
			rand = NULL;
		}
	}

	__declspec(dllexport) int32_t UniformRand(cuda_rand* rand, float *result, int64_t amount_of_numbers) {
		if (amount_of_numbers > rand->max_size()) return marshal_cuda_error(cudaErrorLaunchFailure);
		return marshal_cuda_error(rand->uniform_rand(amount_of_numbers, result));
	}
	__declspec(dllexport) int32_t UniformRandDouble(cuda_rand* rand, double *result, int64_t amount_of_numbers) {
		if (amount_of_numbers > rand->max_size()) return marshal_cuda_error(cudaErrorLaunchFailure);
		return marshal_cuda_error(rand->uniform_rand_double(amount_of_numbers, result));
	}
	__declspec(dllexport) int32_t NormalRand(cuda_rand* rand, float *result, int64_t amount_of_numbers) {
		if (amount_of_numbers > rand->max_size()) return marshal_cuda_error(cudaErrorLaunchFailure);
		return marshal_cuda_error(rand->normal_rand(amount_of_numbers, result));
	}
	__declspec(dllexport) int32_t NormalRandDouble(cuda_rand* rand, double *result, int64_t amount_of_numbers) {
		if (amount_of_numbers > rand->max_size()) return marshal_cuda_error(cudaErrorLaunchFailure);
		return marshal_cuda_error(rand->normal_rand_double(amount_of_numbers, result));
	}
	__declspec(dllexport) int32_t LogNormalRand(cuda_rand* rand, float *result, int64_t amount_of_numbers, float mean, float stddev) {
		if (amount_of_numbers > rand->max_size()) return marshal_cuda_error(cudaErrorLaunchFailure);
		return marshal_cuda_error(rand->log_normal_rand(amount_of_numbers, result, mean, stddev));
	}
	__declspec(dllexport) int32_t LogNormalRandDouble(cuda_rand* rand, double *result, int64_t amount_of_numbers, float mean, float stddev) {
		if (amount_of_numbers > rand->max_size()) return marshal_cuda_error(cudaErrorLaunchFailure);
		return marshal_cuda_error(rand->log_normal_rand_double(amount_of_numbers, result, mean, stddev));
	}
	__declspec(dllexport) int32_t PoissonRand(cuda_rand* rand, int32_t *result, int64_t amount_of_numbers, double lambda) {
		if (amount_of_numbers > rand->max_size()) return marshal_cuda_error(cudaErrorLaunchFailure);
		return marshal_cuda_error(rand->poisson_rand(amount_of_numbers, result, lambda));
	}
}