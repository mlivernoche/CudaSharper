#pragma once
#include "cuda_rand.h"

// sizeof(curandState_t) = 48.
// Launching 64 threads, 48 * 64 = 3072; 3072 * 32 = 98304 bytes. (32 = active warps).
// 98304 is the amount of shared memory per block available on Pascal/Maxwell!
// Using the shared memory for this kernel can halve the execution time (on Pascal).
// For Kepler, we have to double the amount of threads to achieve 100% occupancy.
#define CURAND_NUM_OF_THREADS 64

void cuda_rand::determine_launch_parameters(int32_t* blocks, int32_t* threads, const int64_t array_size, const int32_t max_block_size, const int32_t max_thread_size) {
	if (*blocks * *threads < array_size) {
		if ((*blocks * 2) <= max_block_size) {
			*blocks = (*blocks * 2);
			cuda_rand::determine_launch_parameters(blocks, threads, array_size, max_block_size, max_thread_size);
		}
		else if ((*threads * 2) <= max_block_size) {
			*threads = (*threads * 2);
			cuda_rand::determine_launch_parameters(blocks, threads, array_size, max_block_size, max_thread_size);
		}
		return;
	}
	return;
}

int64_t cuda_rand::time_seed() {
	// time(NULL) is not precise enough to produce different sets of random numbers.
	return std::chrono::system_clock::now().time_since_epoch().count();
}

__global__ void cuda_rand_uniform_rand_kernel(int64_t seed, float *numbers, const int64_t maximum) {
	extern __shared__ curandState_t curandStateShared[];

	int xid = blockIdx.x * blockDim.x + threadIdx.x;
	curand_init(seed + xid, 0, 0, &curandStateShared[threadIdx.x]);

	for (int i = xid; i < maximum; i += blockDim.x * gridDim.x) {
		numbers[i] = curand_uniform(&curandStateShared[threadIdx.x]);
	}
}

cudaError_t cuda_rand::uniform_rand(int32_t device_id, const int64_t amount_of_numbers, float *result) {
	cudaError_t errorCode = cudaSetDevice(device_id);
	if (errorCode != cudaSuccess) return errorCode;

	cudaDeviceProp prop;
	errorCode = cudaGetDeviceProperties(&prop, device_id);
	if (errorCode != cudaSuccess) return errorCode;

	// kernel prefers blocks over threads, but does not like only blocks and no threads.
	int32_t threads = CURAND_NUM_OF_THREADS;
	int32_t maxthreads = CURAND_NUM_OF_THREADS;
	int32_t blocks = 2;

	// sizeof(curandState_t) = 48.
	// Launching 64 threads, 48 * 64 = 3072; 3072 * 32 = 98304 bytes. (32 = block limit).
	// 98304 is the amount of shared memory per block available on Pascal/Maxwell!
	// Using the shared memory for this kernel can halve the execution time (on Pascal).
	// For Kepler, we have to double the amount of threads to achieve 100% occupancy.
	if (prop.major == 3 && prop.minor == 5) {
		threads *= 2;
		maxthreads *= 2;
	}

	// See if we can increase the block size even more.
	// Regarding the max threads, see above.
	cuda_rand::determine_launch_parameters(&blocks, &threads, amount_of_numbers, prop.multiProcessorCount * 32, maxthreads);

	float *d_nums;
	errorCode = cudaMalloc(&d_nums, amount_of_numbers * sizeof(float));
	if (errorCode != cudaSuccess) return errorCode;

	// this kernel loves bandwidth, so distributing resources should be based on used shared memory.
	// 0 = int = offset (the start of the loop), 1 = int = the end of the loop
	size_t sharedMem = sizeof(curandState_t) * threads;
	cuda_rand_uniform_rand_kernel << <blocks, threads, sharedMem >> > (cuda_rand::time_seed(), d_nums, amount_of_numbers);

	errorCode = cudaMemcpy(result, d_nums, amount_of_numbers * sizeof(float), cudaMemcpyDeviceToHost);
	if (errorCode != cudaSuccess) return errorCode;

	errorCode = cudaFree(d_nums);
	if (errorCode != cudaSuccess) return errorCode;

	return cudaSuccess;
}

__global__ void cuda_rand_uniform_rand_double_kernel(int64_t seed, double *numbers, const int64_t maximum) {
	extern __shared__ curandState_t curandStateShared[];

	int xid = blockIdx.x * blockDim.x + threadIdx.x;
	curand_init(seed + xid, 0, 0, &curandStateShared[threadIdx.x]);

	for (int i = xid; i < maximum; i += blockDim.x * gridDim.x) {
		numbers[i] = curand_uniform_double(&curandStateShared[threadIdx.x]);
	}
}

cudaError_t cuda_rand::uniform_rand_double(int32_t device_id, const int64_t amount_of_numbers, double *result) {
	cudaError_t errorCode = cudaSetDevice(device_id);
	if (errorCode != cudaSuccess) return errorCode;

	cudaDeviceProp prop;
	errorCode = cudaGetDeviceProperties(&prop, device_id);
	if (errorCode != cudaSuccess) return errorCode;

	// kernel prefers blocks over threads, but does not like only blocks and no threads.
	int32_t threads = CURAND_NUM_OF_THREADS;
	int32_t maxthreads = CURAND_NUM_OF_THREADS;
	int32_t blocks = 2;

	// sizeof(curandState_t) = 48.
	// Launching 64 threads, 48 * 64 = 3072; 3072 * 32 = 98304 bytes. (32 = block limit).
	// 98304 is the amount of shared memory per block available on Pascal/Maxwell!
	// Using the shared memory for this kernel can halve the execution time (on Pascal).
	// For Kepler, we have to double the amount of threads to achieve 100% occupancy.
	if (prop.major == 3 && prop.minor == 5) {
		threads *= 2;
		maxthreads *= 2;
	}

	// See if we can increase the block size even more.
	// Regarding the max threads, see above.
	cuda_rand::determine_launch_parameters(&blocks, &threads, amount_of_numbers, prop.multiProcessorCount * 32, maxthreads);

	double *d_nums;
	errorCode = cudaMalloc(&d_nums, amount_of_numbers * sizeof(double));
	if (errorCode != cudaSuccess) return errorCode;

	size_t sharedMem = sizeof(curandState_t) * threads;
	cuda_rand_uniform_rand_double_kernel << <blocks, threads, sharedMem >> >(cuda_rand::time_seed(), d_nums, amount_of_numbers);

	errorCode = cudaMemcpy(result, d_nums, amount_of_numbers * sizeof(double), cudaMemcpyDeviceToHost);
	if (errorCode != cudaSuccess) return errorCode;

	errorCode = cudaFree(d_nums);
	if (errorCode != cudaSuccess) return errorCode;

	return cudaSuccess;
}

__global__ void cuda_rand_normal_rand_kernel(int64_t seed, float *numbers, const int64_t maximum) {
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

cudaError_t cuda_rand::normal_rand(int32_t device_id, const int64_t amount_of_numbers, float *result) {
	cudaError_t errorCode = cudaSetDevice(device_id);
	if (errorCode != cudaSuccess) return errorCode;

	cudaDeviceProp prop;
	errorCode = cudaGetDeviceProperties(&prop, device_id);
	if (errorCode != cudaSuccess) return errorCode;

	size_t data_size_in_memory = sizeof(float) * amount_of_numbers;

	// kernel prefers blocks over threads, but does not like only blocks and no threads.
	int32_t threads = CURAND_NUM_OF_THREADS;
	int32_t maxthreads = CURAND_NUM_OF_THREADS;
	int32_t blocks = 2;

	// sizeof(curandState_t) = 48.
	// Launching 64 threads, 48 * 64 = 3072; 3072 * 32 = 98304 bytes. (32 = block limit).
	// 98304 is the amount of shared memory per block available on Pascal/Maxwell!
	// Using the shared memory for this kernel can halve the execution time (on Pascal).
	// For Kepler, we have to double the amount of threads to achieve 100% occupancy.
	if (prop.major == 3 && prop.minor == 5) {
		threads *= 2;
		maxthreads *= 2;
	}

	// See if we can increase the block size even more.
	// Regarding the max threads, see above.
	cuda_rand::determine_launch_parameters(&blocks, &threads, amount_of_numbers, prop.multiProcessorCount * 32, maxthreads);

	float *d_nums;

	errorCode = cudaMalloc(&d_nums, data_size_in_memory);
	if (errorCode != cudaSuccess) return errorCode;

	// this kernel loves bandwidth, so distributing resources should be based on used shared memory.
	// 0 = int = offset (the start of the loop), 1 = int = the end of the loop
	size_t sharedMem = sizeof(curandState_t) * threads;
	cuda_rand_normal_rand_kernel << <blocks, threads, sharedMem >> >(cuda_rand::time_seed(), d_nums, amount_of_numbers);

	errorCode = cudaMemcpy(result, d_nums, data_size_in_memory, cudaMemcpyDeviceToHost);
	if (errorCode != cudaSuccess) return errorCode;

	errorCode = cudaFree(d_nums);
	if (errorCode != cudaSuccess) return errorCode;

	return cudaSuccess;
}

__global__ void cuda_rand_normal_rand_double_kernel(int64_t seed, double *numbers, const int64_t maximum) {
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

cudaError_t cuda_rand::normal_rand_double(int32_t device_id, const int64_t amount_of_numbers, double *result) {
	cudaError_t errorCode = cudaSetDevice(device_id);
	if (errorCode != cudaSuccess) return errorCode;

	cudaDeviceProp prop;
	errorCode = cudaGetDeviceProperties(&prop, device_id);
	if (errorCode != cudaSuccess) return errorCode;

	// kernel prefers blocks over threads, but does not like only blocks and no threads.
	int32_t threads = CURAND_NUM_OF_THREADS;
	int32_t maxthreads = CURAND_NUM_OF_THREADS;
	int32_t blocks = 2;

	// sizeof(curandState_t) = 48.
	// Launching 64 threads, 48 * 64 = 3072; 3072 * 32 = 98304 bytes. (32 = block limit).
	// 98304 is the amount of shared memory per block available on Pascal/Maxwell!
	// Using the shared memory for this kernel can halve the execution time (on Pascal).
	// For Kepler, we have to double the amount of threads to achieve 100% occupancy.
	if (prop.major == 3 && prop.minor == 5) {
		threads *= 2;
		maxthreads *= 2;
	}

	// See if we can increase the block size even more.
	// Regarding the max threads, see above.
	cuda_rand::determine_launch_parameters(&blocks, &threads, amount_of_numbers, prop.multiProcessorCount * 32, maxthreads);

	double *d_nums;
	errorCode = cudaMalloc(&d_nums, amount_of_numbers * sizeof(double));
	if (errorCode != cudaSuccess) return errorCode;

	// this kernel loves bandwidth, so distributing resources should be based on used shared memory.
	// 0 = int = offset (the start of the loop), 1 = int = the end of the loop
	size_t sharedMem = sizeof(curandState_t) * threads;
	cuda_rand_normal_rand_double_kernel << <blocks, threads, sharedMem >> >(cuda_rand::time_seed(), d_nums, amount_of_numbers);

	errorCode = cudaMemcpy(result, d_nums, amount_of_numbers * sizeof(double), cudaMemcpyDeviceToHost);
	if (errorCode != cudaSuccess) return errorCode;

	errorCode = cudaFree(d_nums);
	if (errorCode != cudaSuccess) return errorCode;

	return cudaSuccess;
}

__global__ void cuda_rand_log_normal_rand_kernel(int64_t seed, float *numbers, const int64_t maximum, float mean, float stddev) {
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

cudaError_t cuda_rand::log_normal_rand(int32_t device_id, const int64_t amount_of_numbers, float *result, float mean, float stddev) {
	cudaError_t errorCode = cudaSetDevice(device_id);
	if (errorCode != cudaSuccess) return errorCode;

	cudaDeviceProp prop;
	errorCode = cudaGetDeviceProperties(&prop, device_id);
	if (errorCode != cudaSuccess) return errorCode;

	// kernel prefers blocks over threads, but does not like only blocks and no threads.
	int32_t threads = CURAND_NUM_OF_THREADS;
	int32_t maxthreads = CURAND_NUM_OF_THREADS;
	int32_t blocks = 2;

	// sizeof(curandState_t) = 48.
	// Launching 64 threads, 48 * 64 = 3072; 3072 * 32 = 98304 bytes. (32 = block limit).
	// 98304 is the amount of shared memory per block available on Pascal/Maxwell!
	// Using the shared memory for this kernel can halve the execution time (on Pascal).
	// For Kepler, we have to double the amount of threads to achieve 100% occupancy.
	if (prop.major == 3 && prop.minor == 5) {
		threads *= 2;
		maxthreads *= 2;
	}

	// See if we can increase the block size even more.
	// Regarding the max threads, see above.
	cuda_rand::determine_launch_parameters(&blocks, &threads, amount_of_numbers, prop.multiProcessorCount * 32, maxthreads);

	float *d_nums;
	errorCode = cudaMalloc(&d_nums, amount_of_numbers * sizeof(float));
	if (errorCode != cudaSuccess) return errorCode;

	// this kernel loves bandwidth, so distributing resources should be based on used shared memory.
	// 0 = int = offset (the start of the loop), 1 = int = the end of the loop
	size_t sharedMem = sizeof(curandState_t) * threads;
	cuda_rand_log_normal_rand_kernel << <blocks, threads, sharedMem >> >(cuda_rand::time_seed(), d_nums, amount_of_numbers, mean, stddev);

	errorCode = cudaMemcpy(result, d_nums, amount_of_numbers * sizeof(float), cudaMemcpyDeviceToHost);
	if (errorCode != cudaSuccess) return errorCode;

	errorCode = cudaFree(d_nums);
	if (errorCode != cudaSuccess) return errorCode;

	return cudaSuccess;
}

__global__ void cuda_rand_log_normal_rand_double_kernel(int64_t seed, double *numbers, const int64_t maximum, double mean, double stddev) {
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

cudaError_t cuda_rand::log_normal_rand_double(int32_t device_id, const int64_t amount_of_numbers, double *result, double mean, double stddev) {
	cudaError_t errorCode = cudaSetDevice(device_id);
	if (errorCode != cudaSuccess) return errorCode;

	cudaDeviceProp prop;
	errorCode = cudaGetDeviceProperties(&prop, device_id);
	if (errorCode != cudaSuccess) return errorCode;

	// kernel prefers blocks over threads, but does not like only blocks and no threads.
	int32_t threads = CURAND_NUM_OF_THREADS;
	int32_t maxthreads = CURAND_NUM_OF_THREADS;
	int32_t blocks = 2;

	// sizeof(curandState_t) = 48.
	// Launching 64 threads, 48 * 64 = 3072; 3072 * 32 = 98304 bytes. (32 = block limit).
	// 98304 is the amount of shared memory per block available on Pascal/Maxwell!
	// Using the shared memory for this kernel can halve the execution time (on Pascal).
	// For Kepler, we have to double the amount of threads to achieve 100% occupancy.
	if (prop.major == 3 && prop.minor == 5) {
		threads *= 2;
		maxthreads *= 2;
	}

	// See if we can increase the block size even more.
	// Regarding the max threads, see above.
	cuda_rand::determine_launch_parameters(&blocks, &threads, amount_of_numbers, prop.multiProcessorCount * 32, maxthreads);

	double *d_nums;
	errorCode = cudaMalloc(&d_nums, amount_of_numbers * sizeof(double));
	if (errorCode != cudaSuccess) return errorCode;

	// this kernel loves bandwidth, so distributing resources should be based on used shared memory.
	// 0 = int = offset (the start of the loop), 1 = int = the end of the loop
	size_t sharedMem = sizeof(curandState_t) * threads;
	cuda_rand_log_normal_rand_double_kernel << <blocks, threads, sharedMem >> >(cuda_rand::time_seed(), d_nums, amount_of_numbers, mean, stddev);

	errorCode = cudaMemcpy(result, d_nums, amount_of_numbers * sizeof(double), cudaMemcpyDeviceToHost);
	if (errorCode != cudaSuccess) return errorCode;

	errorCode = cudaFree(d_nums);
	if (errorCode != cudaSuccess) return errorCode;

	return cudaSuccess;
}

__global__ void cuda_rand_poisson_rand_kernel(int64_t seed, int32_t *numbers, const int64_t maximum, double lambda) {
	extern __shared__ curandState_t curandStateShared[];

	int xid = blockIdx.x * blockDim.x + threadIdx.x;
	curand_init(seed + xid, 0, 0, &curandStateShared[threadIdx.x]);

	for (int i = xid; i < maximum; i += blockDim.x * gridDim.x) {
		numbers[i] = curand_poisson(&curandStateShared[threadIdx.x], lambda);
	}
}

cudaError_t cuda_rand::poisson_rand(int32_t device_id, const int64_t amount_of_numbers, int32_t *result, double lambda) {
	cudaError_t errorCode = cudaSetDevice(device_id);
	if (errorCode != cudaSuccess) return errorCode;

	cudaDeviceProp prop;
	errorCode = cudaGetDeviceProperties(&prop, device_id);
	if (errorCode != cudaSuccess) return errorCode;

	// kernel prefers blocks over threads, but does not like only blocks and no threads.
	int32_t threads = CURAND_NUM_OF_THREADS;
	int32_t maxthreads = CURAND_NUM_OF_THREADS;
	int32_t blocks = 2;

	// sizeof(curandState_t) = 48.
	// Launching 64 threads, 48 * 64 = 3072; 3072 * 32 = 98304 bytes. (32 = block limit).
	// 98304 is the amount of shared memory per block available on Pascal/Maxwell!
	// Using the shared memory for this kernel can halve the execution time (on Pascal).
	// For Kepler, we have to double the amount of threads to achieve 100% occupancy.
	if (prop.major == 3 && prop.minor == 5) {
		threads *= 2;
		maxthreads *= 2;
	}

	// See if we can increase the block size even more.
	// Regarding the max threads, see above.
	cuda_rand::determine_launch_parameters(&blocks, &threads, amount_of_numbers, prop.multiProcessorCount * 32, maxthreads);

	int *d_nums;
	errorCode = cudaMalloc(&d_nums, amount_of_numbers * sizeof(int));
	if (errorCode != cudaSuccess) return errorCode;

	// this kernel loves bandwidth, so distributing resources should be based on used shared memory.
	size_t sharedMem = sizeof(curandState_t) * threads;
	cuda_rand_poisson_rand_kernel << <blocks, threads, sharedMem >> > (cuda_rand::time_seed(), d_nums, amount_of_numbers, lambda);

	errorCode = cudaMemcpy(result, d_nums, amount_of_numbers * sizeof(float), cudaMemcpyDeviceToHost);
	if (errorCode != cudaSuccess) return errorCode;

	errorCode = cudaFree(d_nums);
	if (errorCode != cudaSuccess) return errorCode;

	return cudaSuccess;
}

extern "C" {
	__declspec(dllexport) int32_t UniformRand(int32_t device_id, float *result, int64_t amount_of_numbers) {
		return marshal_cuda_error(cuda_rand::uniform_rand(device_id, amount_of_numbers, result));
	}
	__declspec(dllexport) int32_t UniformRandDouble(int32_t device_id, double *result, int64_t amount_of_numbers) {
		return marshal_cuda_error(cuda_rand::uniform_rand_double(device_id, amount_of_numbers, result));
	}
	__declspec(dllexport) int32_t NormalRand(int32_t device_id, float *result, int64_t amount_of_numbers) {
		return marshal_cuda_error(cuda_rand::normal_rand(device_id, amount_of_numbers, result));
	}
	__declspec(dllexport) int32_t NormalRandDouble(int32_t device_id, double *result, int64_t amount_of_numbers) {
		return marshal_cuda_error(cuda_rand::normal_rand_double(device_id, amount_of_numbers, result));
	}
	__declspec(dllexport) int32_t LogNormalRand(int32_t device_id, float *result, int64_t amount_of_numbers, float mean, float stddev) {
		return marshal_cuda_error(cuda_rand::log_normal_rand(device_id, amount_of_numbers, result, mean, stddev));
	}
	__declspec(dllexport) int32_t LogNormalRandDouble(int32_t device_id, double *result, int64_t amount_of_numbers, float mean, float stddev) {
		return marshal_cuda_error(cuda_rand::log_normal_rand_double(device_id, amount_of_numbers, result, mean, stddev));
	}
	__declspec(dllexport) int32_t PoissonRand(int32_t device_id, int32_t *result, int64_t amount_of_numbers, double lambda) {
		return marshal_cuda_error(cuda_rand::poisson_rand(device_id, amount_of_numbers, result, lambda));
	}
}