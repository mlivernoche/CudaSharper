#pragma once
#include "cuda_rand.h"

// sizeof(curandState_t) = 48.
// Launching 64 threads, 48 * 64 = 3072; 3072 * 32 = 98304 bytes. (32 = block limit).
// 98304 is the amount of shared memory per block available on Pascal/Maxwell!
// Using the shared memory for this kernel can halve the execution time (on Pascal).
// For Kepler, we have to double the amount of threads to achieve 100% occupancy.
#define CURAND_NUM_OF_THREADS 64

// This has to be a multiple of 2.
#define CURAND_MIN_SIZE_PER_THREAD 16

void cuda_rand_determine_launch_parameters(unsigned long int* blocks, unsigned long int* threads, unsigned long int* number_per_thread, unsigned long int max_block_size, unsigned long int max_thread_size) {
	if (*number_per_thread > CURAND_MIN_SIZE_PER_THREAD)
	{
		if ((*blocks * 2) < max_block_size)
		{
			*blocks = (*blocks * 2);
			*number_per_thread = (int)ceil(*number_per_thread / 2) + 1;
			cuda_rand_determine_launch_parameters(blocks, threads, number_per_thread, max_block_size, max_thread_size);
		}
		else if ((*threads * 2) < max_thread_size)
		{
			*threads = (*threads * 2);
			*number_per_thread = (int)ceil(*number_per_thread / 2) + 1;
			cuda_rand_determine_launch_parameters(blocks, threads, number_per_thread, max_block_size, max_thread_size);
		}
		return;
	}
	return;
}

long long int cuda_rand_time_seed() {
	// time(NULL) is not precise enough to produce different sets of random numbers.
	return std::chrono::system_clock::now().time_since_epoch().count();
}

extern "C" __declspec(dllexport) void UniformRand(unsigned int device_id, unsigned int amount_of_numbers, float *result) {
	cuda_rand_uniform_rand(device_id, amount_of_numbers, result);
}

extern "C" __declspec(dllexport) void UniformRandDouble(unsigned int device_id, unsigned int amount_of_numbers, double *result) {
	cuda_rand_uniform_rand_double(device_id, amount_of_numbers, result);
}

extern "C" __declspec(dllexport) void NormalRand(unsigned int device_id, unsigned int amount_of_numbers, float *result) {
	cuda_rand_normal_rand(device_id, amount_of_numbers, result);
}

extern "C" __declspec(dllexport) void NormalRandDouble(unsigned int device_id, unsigned int amount_of_numbers, double *result) {
	cuda_rand_normal_rand_double(device_id, amount_of_numbers, result);
}

extern "C" __declspec(dllexport) void LogNormalRand(unsigned int device_id, unsigned int amount_of_numbers, float *result, float mean, float stddev) {
	cuda_rand_log_normal_rand(device_id, amount_of_numbers, result, mean, stddev);
}

extern "C" __declspec(dllexport) void LogNormalRandDouble(unsigned int device_id, unsigned int amount_of_numbers, double *result, float mean, float stddev) {
	cuda_rand_log_normal_rand_double(device_id, amount_of_numbers, result, mean, stddev);
}

extern "C" __declspec(dllexport) void PoissonRand(unsigned int device_id, unsigned int amount_of_numbers, int *result, double lambda) {
	cuda_rand_poisson_rand(device_id, amount_of_numbers, result, lambda);
}

__global__ void cuda_rand_uniform_rand_kernel(long long int seed, float *numbers, unsigned int count, unsigned int maximum) {
	extern __shared__ int smem[];
	curandState_t *curandStateShared = (curandState_t*)&smem[0];

	int xid = (threadIdx.x + (blockIdx.x * blockDim.x));

	curand_init(seed + xid, 0, 0, &curandStateShared[threadIdx.x]);

	// This is the starting point of the array that this kernel is responsible for.
	int kernel_block = (xid * count);

	if (count + kernel_block < maximum) {
		// We can do the entire chunk of numbers in the array.
		for (int n = 0; n < count; n++) {
			numbers[n + kernel_block] = curand_uniform(&curandStateShared[threadIdx.x]);
		}
	}
	else if (kernel_block < maximum) {
		// We can't do the entire chunk of numbers in the array, we can still do some of it.
		for (int n = 0; n < maximum - kernel_block; n++) {
			numbers[n + kernel_block] = curand_uniform(&curandStateShared[threadIdx.x]);
		}
	}
}

void cuda_rand_uniform_rand(unsigned int device_id, unsigned int amount_of_numbers, float *result) {
	cudaError_t gpu_device = cudaSetDevice(device_id);

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, device_id);

	// kernel prefers blocks over threads, but does not like only blocks and no threads.
	unsigned long int threads = CURAND_NUM_OF_THREADS;
	unsigned long int blocks = 2;

	// sizeof(curandState_t) = 48.
	// Launching 64 threads, 48 * 64 = 3072; 3072 * 32 = 98304 bytes. (32 = block limit).
	// 98304 is the amount of shared memory per block available on Pascal/Maxwell!
	// Using the shared memory for this kernel can halve the execution time (on Pascal).
	// For Kepler, we have to double the amount of threads to achieve 100% occupancy.
	if (prop.major == 3 && prop.minor == 5) {
		threads = threads * 2;
	}

	// Figure out how many numbers each thread will have to generate.
	unsigned long int numberPerThread = (amount_of_numbers / (blocks * threads)) + 1;

	// See if we can increase the block size even more.
	// Regarding the max threads, see above.
	cuda_rand_determine_launch_parameters(&blocks, &threads, &numberPerThread, prop.maxGridSize[0], prop.maxThreadsDim[0]);

	float *d_nums;
	cudaMalloc(&d_nums, amount_of_numbers * sizeof(float));

	// this kernel loves bandwidth, so distributing resources should be based on used shared memory.
	// 0 = int = offset (the start of the loop), 1 = int = the end of the loop
	size_t sharedMem = sizeof(curandState_t) * threads;
	cuda_rand_uniform_rand_kernel << <blocks, threads, sharedMem >> > (cuda_rand_time_seed(), d_nums, numberPerThread, amount_of_numbers);

	cudaMemcpy(result, d_nums, amount_of_numbers * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(d_nums);
}

__global__ void cuda_rand_uniform_rand_double_kernel(long long int seed, double *numbers, unsigned int count, unsigned int maximum) {
	extern __shared__ int smem[];
	curandState_t *curandStateShared = (curandState_t*)&smem[0];

	int xid = (threadIdx.x + (blockIdx.x * blockDim.x));

	curand_init(seed + xid, 0, 0, &curandStateShared[threadIdx.x]);

	// This is the starting point of the array that this kernel is responsible for.
	int kernel_block = (xid * count);

	if (count + kernel_block < maximum) {
		// We can do the entire chunk of numbers in the array.
		for (int n = 0; n < count; n++) {
			numbers[n + kernel_block] = curand_uniform_double(&curandStateShared[threadIdx.x]);
		}
	}
	else if (kernel_block < maximum) {
		// We can't do the entire chunk of numbers in the array, we can still do some of it.
		for (int n = 0; n < maximum - kernel_block; n++) {
			numbers[n + kernel_block] = curand_uniform_double(&curandStateShared[threadIdx.x]);
		}
	}
}

void cuda_rand_uniform_rand_double(unsigned int device_id, unsigned int amount_of_numbers, double *result) {
	cudaError_t gpu_device = cudaSetDevice(device_id);

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, device_id);

	// kernel prefers blocks over threads, but does not like only blocks and no threads.
	unsigned long int threads = CURAND_NUM_OF_THREADS;
	unsigned long int blocks = 2;

	// sizeof(curandState_t) = 48.
	// Launching 64 threads, 48 * 64 = 3072; 3072 * 32 = 98304 bytes. (32 = block limit).
	// 98304 is the amount of shared memory per block available on Pascal/Maxwell!
	// Using the shared memory for this kernel can halve the execution time (on Pascal).
	// For Kepler, we have to double the amount of threads to achieve 100% occupancy.
	if (prop.major == 3 && prop.minor == 5) {
		threads = threads * 2;
	}

	// Figure out how many numbers each thread will have to generate.
	unsigned long int numberPerThread = (amount_of_numbers / (blocks * threads)) + 1;

	// See if we can increase the block size even more.
	// Regarding the max threads, see above.
	cuda_rand_determine_launch_parameters(&blocks, &threads, &numberPerThread, prop.maxGridSize[0], prop.maxThreadsDim[0]);

	double *d_nums;
	cudaMalloc(&d_nums, amount_of_numbers * sizeof(double));

	size_t sharedMem = sizeof(curandState_t) * threads;
	cuda_rand_uniform_rand_double_kernel << <blocks, threads, sharedMem >> >(cuda_rand_time_seed(), d_nums, numberPerThread, amount_of_numbers);

	cudaMemcpy(result, d_nums, amount_of_numbers * sizeof(double), cudaMemcpyDeviceToHost);

	cudaFree(d_nums);
}

__global__ void cuda_rand_normal_rand_kernel(long long int seed, float *numbers, unsigned int count, unsigned int maximum) {
	// (threadIdx.x * N) + + offset
	// N = number of elements
	// offset = the position of the desired element.
	// 0 = int = offset (the start of the loop), 1 = int = the end of the loop
	extern __shared__ int smem[];
	curandState_t *curandStateShared = (curandState_t*)&smem[0];

	// The state.
	int xid = (threadIdx.x + (blockIdx.x * blockDim.x));
	
	curand_init(seed + xid, 0, 0, &curandStateShared[threadIdx.x]);

	// This is the starting point of the array that this kernel is responsible for.
	int kernel_block = (xid * count);

	if (count + kernel_block < maximum) {
		// We can do the entire chunk of numbers in the array.
		for (int n = 0; n < count; n++) {
			numbers[n + kernel_block] = curand_normal(&curandStateShared[threadIdx.x]);
		}
	}
	else if (kernel_block < maximum) {
		// We can't do the entire chunk of numbers in the array, we can still do some of it.
		for (int n = 0; n < maximum - kernel_block; n++) {
			numbers[n + kernel_block] = curand_normal(&curandStateShared[threadIdx.x]);
		}
	}
}

void cuda_rand_normal_rand(unsigned int device_id, unsigned int amount_of_numbers, float *result) {
	cudaError_t gpu_device = cudaSetDevice(device_id);

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, device_id);

	// kernel prefers blocks over threads, but does not like only blocks and no threads.
	unsigned long int threads = CURAND_NUM_OF_THREADS;
	unsigned long int blocks = 2;

	// sizeof(curandState_t) = 48.
	// Launching 64 threads, 48 * 64 = 3072; 3072 * 32 = 98304 bytes. (32 = block limit).
	// 98304 is the amount of shared memory per block available on Pascal/Maxwell!
	// Using the shared memory for this kernel can halve the execution time (on Pascal).
	// For Kepler, we have to double the amount of threads to achieve 100% occupancy.
	if (prop.major == 3 && prop.minor == 5) {
		threads = threads * 2;
	}

	// Figure out how many numbers each thread will have to generate.
	unsigned long int numberPerThread = (amount_of_numbers / (blocks * threads)) + 1;

	// See if we can increase the block size even more.
	// Regarding the max threads, see above.
	cuda_rand_determine_launch_parameters(&blocks, &threads, &numberPerThread, prop.maxGridSize[0], prop.maxThreadsDim[0]);

	float *d_nums;
	cudaMalloc(&d_nums, amount_of_numbers * sizeof(float));

	// this kernel loves bandwidth, so distributing resources should be based on used shared memory.
	// 0 = int = offset (the start of the loop), 1 = int = the end of the loop
	size_t sharedMem = sizeof(curandState_t) * threads;
	cuda_rand_normal_rand_kernel << <blocks, threads, sharedMem >> >(cuda_rand_time_seed(), d_nums, numberPerThread, amount_of_numbers);

	cudaMemcpy(result, d_nums, amount_of_numbers * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(d_nums);
}

__global__ void cuda_rand_normal_rand_double_kernel(long long int seed, double *numbers, unsigned int count, unsigned int maximum) {
	extern __shared__ int smem[];
	curandState_t *curandStateShared = (curandState_t*)&smem[0];

	int xid = (threadIdx.x + (blockIdx.x * blockDim.x));

	curand_init(seed + xid, 0, 0, &curandStateShared[threadIdx.x]);

	// This is the starting point of the array that this kernel is responsible for.
	int kernel_block = (xid * count);

	if (count + kernel_block < maximum) {
		// We can do the entire chunk of numbers in the array.
		for (int n = 0; n < count; n++) {
			numbers[n + kernel_block] = curand_normal_double(&curandStateShared[threadIdx.x]);
		}
	}
	else if (kernel_block < maximum) {
		// We can't do the entire chunk of numbers in the array, we can still do some of it.
		for (int n = 0; n < maximum - kernel_block; n++) {
			numbers[n + kernel_block] = curand_normal_double(&curandStateShared[threadIdx.x]);
		}
	}
}

void cuda_rand_normal_rand_double(unsigned int device_id, unsigned int amount_of_numbers, double *result) {
	cudaError_t gpu_device = cudaSetDevice(device_id);

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, device_id);

	// kernel prefers blocks over threads, but does not like only blocks and no threads.
	unsigned long int threads = CURAND_NUM_OF_THREADS;
	unsigned long int blocks = 2;

	// sizeof(curandState_t) = 48.
	// Launching 64 threads, 48 * 64 = 3072; 3072 * 32 = 98304 bytes. (32 = block limit).
	// 98304 is the amount of shared memory per block available on Pascal/Maxwell!
	// Using the shared memory for this kernel can halve the execution time (on Pascal).
	// For Kepler, we have to double the amount of threads to achieve 100% occupancy.
	if (prop.major == 3 && prop.minor == 5) {
		threads = threads * 2;
	}

	// Figure out how many numbers each thread will have to generate.
	unsigned long int numberPerThread = (amount_of_numbers / (blocks * threads)) + 1;

	// See if we can increase the block size even more.
	// Regarding the max threads, see above.
	cuda_rand_determine_launch_parameters(&blocks, &threads, &numberPerThread, prop.maxGridSize[0], prop.maxThreadsDim[0]);

	double *d_nums;
	cudaMalloc(&d_nums, amount_of_numbers * sizeof(double));

	// this kernel loves bandwidth, so distributing resources should be based on used shared memory.
	// 0 = int = offset (the start of the loop), 1 = int = the end of the loop
	size_t sharedMem = sizeof(curandState_t) * threads;
	cuda_rand_normal_rand_double_kernel << <blocks, threads, sharedMem >> >(cuda_rand_time_seed(), d_nums, numberPerThread, amount_of_numbers);

	cudaMemcpy(result, d_nums, amount_of_numbers * sizeof(double), cudaMemcpyDeviceToHost);

	cudaFree(d_nums);
}

__global__ void cuda_rand_log_normal_rand_kernel(long long int seed, float *numbers, unsigned int count, unsigned int maximum, float mean, float stddev) {
	extern __shared__ int smem[];
	curandState_t *curandStateShared = (curandState_t*)&smem[0];

	int xid = (threadIdx.x + (blockIdx.x * blockDim.x));

	curand_init(seed + xid, 0, 0, &curandStateShared[threadIdx.x]);

	// This is the starting point of the array that this kernel is responsible for.
	int kernel_block = (xid * count);

	if (count + kernel_block < maximum) {
		// We can do the entire chunk of numbers in the array.
		for (int n = 0; n < count; n++) {
			numbers[n + kernel_block] = curand_log_normal(&curandStateShared[threadIdx.x], mean, stddev);
		}
	}
	else if (kernel_block < maximum) {
		// We can't do the entire chunk of numbers in the array, we can still do some of it.
		for (int n = 0; n < maximum - kernel_block; n++) {
			numbers[n + kernel_block] = curand_log_normal(&curandStateShared[threadIdx.x], mean, stddev);
		}
	}
}

void cuda_rand_log_normal_rand(unsigned int device_id, unsigned int amount_of_numbers, float *result, float mean, float stddev) {
	cudaError_t gpu_device = cudaSetDevice(device_id);

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, device_id);

	// kernel prefers blocks over threads, but does not like only blocks and no threads.
	unsigned long int threads = CURAND_NUM_OF_THREADS;
	unsigned long int blocks = 2;

	// sizeof(curandState_t) = 48.
	// Launching 64 threads, 48 * 64 = 3072; 3072 * 32 = 98304 bytes. (32 = block limit).
	// 98304 is the amount of shared memory per block available on Pascal/Maxwell!
	// Using the shared memory for this kernel can halve the execution time (on Pascal).
	// For Kepler, we have to double the amount of threads to achieve 100% occupancy.
	if (prop.major == 3 && prop.minor == 5) {
		threads = threads * 2;
	}

	// Figure out how many numbers each thread will have to generate.
	unsigned long int numberPerThread = (amount_of_numbers / (blocks * threads)) + 1;

	// See if we can increase the block size even more.
	// Regarding the max threads, see above.
	cuda_rand_determine_launch_parameters(&blocks, &threads, &numberPerThread, prop.maxGridSize[0], prop.maxThreadsDim[0]);

	float *d_nums;
	cudaMalloc(&d_nums, amount_of_numbers * sizeof(float));

	// this kernel loves bandwidth, so distributing resources should be based on used shared memory.
	// 0 = int = offset (the start of the loop), 1 = int = the end of the loop
	size_t sharedMem = sizeof(curandState_t) * threads;
	cuda_rand_log_normal_rand_kernel << <blocks, threads, sharedMem >> >(cuda_rand_time_seed(), d_nums, numberPerThread, amount_of_numbers, mean, stddev);

	cudaMemcpy(result, d_nums, amount_of_numbers * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(d_nums);
}

__global__ void cuda_rand_log_normal_rand_double_kernel(long long int seed, double *numbers, unsigned int count, unsigned int maximum, double mean, double stddev) {
	extern __shared__ int smem[];
	curandState_t *curandStateShared = (curandState_t*)&smem[0];

	int xid = (threadIdx.x + (blockIdx.x * blockDim.x));

	curand_init(seed + xid, 0, 0, &curandStateShared[threadIdx.x]);

	// This is the starting point of the array that this kernel is responsible for.
	int kernel_block = (xid * count);

	if (count + kernel_block < maximum) {
		// We can do the entire chunk of numbers in the array.
		for (int n = 0; n < count; n++) {
			numbers[n + kernel_block] = curand_log_normal_double(&curandStateShared[threadIdx.x], mean, stddev);
		}
	}
	else if (kernel_block < maximum) {
		// We can't do the entire chunk of numbers in the array, we can still do some of it.
		for (int n = 0; n < maximum - kernel_block; n++) {
			numbers[n + kernel_block] = curand_log_normal_double(&curandStateShared[threadIdx.x], mean, stddev);
		}
	}
}

void cuda_rand_log_normal_rand_double(unsigned int device_id, unsigned int amount_of_numbers, double *result, double mean, double stddev) {
	cudaError_t gpu_device = cudaSetDevice(device_id);

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, device_id);

	// kernel prefers blocks over threads, but does not like only blocks and no threads.
	unsigned long int threads = CURAND_NUM_OF_THREADS;
	unsigned long int blocks = 2;

	// sizeof(curandState_t) = 48.
	// Launching 64 threads, 48 * 64 = 3072; 3072 * 32 = 98304 bytes. (32 = block limit).
	// 98304 is the amount of shared memory per block available on Pascal/Maxwell!
	// Using the shared memory for this kernel can halve the execution time (on Pascal).
	// For Kepler, we have to double the amount of threads to achieve 100% occupancy.
	if (prop.major == 3 && prop.minor == 5) {
		threads = threads * 2;
	}

	// Figure out how many numbers each thread will have to generate.
	unsigned long int numberPerThread = (amount_of_numbers / (blocks * threads)) + 1;

	// See if we can increase the block size even more.
	// Regarding the max threads, see above.
	cuda_rand_determine_launch_parameters(&blocks, &threads, &numberPerThread, prop.maxGridSize[0], prop.maxThreadsDim[0]);

	double *d_nums;
	cudaMalloc(&d_nums, amount_of_numbers * sizeof(double));

	// this kernel loves bandwidth, so distributing resources should be based on used shared memory.
	// 0 = int = offset (the start of the loop), 1 = int = the end of the loop
	size_t sharedMem = sizeof(curandState_t) * threads;
	cuda_rand_log_normal_rand_double_kernel << <blocks, threads, sharedMem >> >(cuda_rand_time_seed(), d_nums, numberPerThread, amount_of_numbers, mean, stddev);

	cudaMemcpy(result, d_nums, amount_of_numbers * sizeof(double), cudaMemcpyDeviceToHost);

	cudaFree(d_nums);
}

__global__ void cuda_rand_poisson_rand_kernel(long long int seed, int *numbers, unsigned int count, unsigned int maximum, double lambda) {
	extern __shared__ int smem[];
	curandState_t *curandStateShared = (curandState_t*)&smem[0];

	int xid = (threadIdx.x + (blockIdx.x * blockDim.x));

	curand_init(seed + xid, 0, 0, &curandStateShared[threadIdx.x]);

	// This is the starting point of the array that this kernel is responsible for.
	int kernel_block = (xid * count);

	if (count + kernel_block < maximum) {
		// We can do the entire chunk of numbers in the array.
		for (int n = 0; n < count; n++) {
			numbers[n + kernel_block] = curand_poisson(&curandStateShared[threadIdx.x], lambda);
		}
	}
	else if (kernel_block < maximum) {
		// We can't do the entire chunk of numbers in the array, we can still do some of it.
		for (int n = 0; n < maximum - kernel_block; n++) {
			numbers[n + kernel_block] = curand_poisson(&curandStateShared[threadIdx.x], lambda);
		}
	}
}

void cuda_rand_poisson_rand(unsigned int device_id, unsigned int amount_of_numbers, int *result, double lambda) {
	cudaError_t gpu_device = cudaSetDevice(device_id);

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, device_id);

	// kernel prefers blocks over threads, but does not like only blocks and no threads.
	unsigned long int threads = CURAND_NUM_OF_THREADS;
	unsigned long int blocks = 2;

	// sizeof(curandState_t) = 48.
	// Launching 64 threads, 48 * 64 = 3072; 3072 * 32 = 98304 bytes. (32 = block limit).
	// 98304 is the amount of shared memory per block available on Pascal/Maxwell!
	// Using the shared memory for this kernel can halve the execution time (on Pascal).
	// For Kepler, we have to double the amount of threads to achieve 100% occupancy.
	if (prop.major == 3 && prop.minor == 5) {
		threads = threads * 2;
	}

	// Figure out how many numbers each thread will have to generate.
	unsigned long int numberPerThread = (amount_of_numbers / (blocks * threads)) + 1;

	// See if we can increase the block size even more.
	// Regarding the max threads, see above.
	cuda_rand_determine_launch_parameters(&blocks, &threads, &numberPerThread, prop.maxGridSize[0], prop.maxThreadsDim[0]);

	int *d_nums;
	cudaMalloc(&d_nums, amount_of_numbers * sizeof(int));

	// this kernel loves bandwidth, so distributing resources should be based on used shared memory.
	size_t sharedMem = sizeof(curandState_t) * threads;
	cuda_rand_poisson_rand_kernel << <blocks, threads, sharedMem >> > (cuda_rand_time_seed(), d_nums, numberPerThread, amount_of_numbers, lambda);

	cudaMemcpy(result, d_nums, amount_of_numbers * sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(d_nums);
}
