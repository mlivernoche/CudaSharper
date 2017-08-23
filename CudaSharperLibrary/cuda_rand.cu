#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_profiler_api.h"
#include "curand.h"
#include "curand_kernel.h"
#include <iostream>
#include <algorithm>
#include <time.h>
#include <chrono>

#include "cuda_rand.h"

// Should be multiple of 32. These kernels tend to prefer more blocks over threads; however, there has to be enough threads to keep the blocks busy.
#define CURAND_NUM_OF_THREADS 32

long long int time_seed() {
	// time(NULL) is not precise enough to produce different sets of random numbers.
	return std::chrono::system_clock::now().time_since_epoch().count();
}

__global__ void init(long long int seed, curandState_t* states) {
	curand_init(
		(threadIdx.x + (blockIdx.x * blockDim.x)) + seed,
		0,
		0,
		&states[(threadIdx.x + (blockIdx.x * blockDim.x))]
	);
}

__global__ void uniform_rand_kernel(curandState_t *states, float *numbers, unsigned int count, unsigned int maximum) {
	// (threadIdx.x * N) + + offset
	// N = number of elements
	// offset = the position of the desired element.
	// 0 = int = offset (the start of the loop), 1 = int = the end of the loop
	extern __shared__ int smem[];

	// unsigned int sharedMem = (sizeof(int) * 2) + ((sizeof(curandState_t) + sizeof(int)) * CURAND_NUM_OF_THREADS);
	int *maximumShared = (int*)&smem[0];
	int *countShared = (int*)&maximumShared[1];
	curandState_t *curandStateShared = (curandState_t*)&countShared[1];
	int *startShared = (int*)&curandStateShared[CURAND_NUM_OF_THREADS];

	if (threadIdx.x == 0) {
		// Make sure we do not go over this.
		maximumShared[0] = maximum;

		// This is the ending point in the *numbers array.
		countShared[0] = count;
	}

	__syncthreads();

	// The state.
	int xid = (threadIdx.x + (blockIdx.x * blockDim.x));
	curandStateShared[threadIdx.x] = states[xid];

	// This is the starting point in the *numbers array.
	startShared[threadIdx.x] = ((threadIdx.x + (blockIdx.x * blockDim.x)) * countShared[0]);

	for (int n = 0; n < countShared[0]; n++) {
		if ((n + startShared[threadIdx.x]) < maximumShared[0]) {
			numbers[(n + startShared[threadIdx.x])] = curand_uniform(&curandStateShared[threadIdx.x]);
		}
	}

	states[xid] = curandStateShared[threadIdx.x];
}

void _uniformRand(int device_id, int amount_of_numbers, float *result) {
	cudaError_t gpu_device = cudaSetDevice(device_id);

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, device_id);

	// kernel prefers blocks over threads, but does not like only blocks and no threads.
	const int blocks = prop.multiProcessorCount * 2;
	const int numberPerThread = (amount_of_numbers / (blocks * CURAND_NUM_OF_THREADS)) + 1;

	curandState_t *states;
	float *d_nums;

	cudaMalloc(&states, blocks * CURAND_NUM_OF_THREADS * sizeof(curandState_t));
	cudaMalloc(&d_nums, amount_of_numbers * sizeof(float));

	init << <blocks, CURAND_NUM_OF_THREADS >> > (time_seed(), states);

	// this kernel loves bandwidth, so distributing resources should be based on used shared memory.
	// 0 = int = offset (the start of the loop), 1 = int = the end of the loop
	unsigned int sharedMem = (sizeof(int) * 2) + ((sizeof(curandState_t) + sizeof(int)) * CURAND_NUM_OF_THREADS);
	uniform_rand_kernel << <blocks, CURAND_NUM_OF_THREADS, sharedMem >> > (states, d_nums, numberPerThread, amount_of_numbers);

	cudaMemcpy(result, d_nums, amount_of_numbers * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(states);
	cudaFree(d_nums);
}

extern "C" __declspec(dllexport) void UniformRand(int device_id, int amount_of_numbers, float *result) {
	_uniformRand(device_id, amount_of_numbers, result);
}

__global__ void uniform_rand_double_kernel(curandState_t *states, double *numbers, unsigned int count, unsigned int maximum) {
	// (threadIdx.x * N) + + offset
	// N = number of elements
	// offset = the position of the desired element.
	// 0 = int = offset (the start of the loop), 1 = int = the end of the loop
	extern __shared__ int smem[];

	// unsigned int sharedMem = (sizeof(int) * 2) + ((sizeof(curandState_t) + sizeof(int)) * CURAND_NUM_OF_THREADS);
	int *maximumShared = (int*)&smem[0];
	int *countShared = (int*)&maximumShared[1];
	curandState_t *curandStateShared = (curandState_t*)&countShared[1];
	int *startShared = (int*)&curandStateShared[CURAND_NUM_OF_THREADS];

	if (threadIdx.x == 0) {
		// Make sure we do not go over this.
		maximumShared[0] = maximum;

		// This is the ending point in the *numbers array.
		countShared[0] = count;
	}

	__syncthreads();

	// The state.
	int xid = (threadIdx.x + (blockIdx.x * blockDim.x));
	curandStateShared[threadIdx.x] = states[xid];

	// This is the starting point in the *numbers array.
	startShared[threadIdx.x] = ((threadIdx.x + (blockIdx.x * blockDim.x)) * countShared[0]);

	for (int n = 0; n < countShared[0]; n++) {
		if ((n + startShared[threadIdx.x]) < maximumShared[0]) {
			numbers[(n + startShared[threadIdx.x])] = curand_uniform_double(&curandStateShared[threadIdx.x]);
		}
	}

	states[xid] = curandStateShared[threadIdx.x];
}

void _uniformRandDouble(int device_id, int amount_of_numbers, double *result) {
	cudaError_t gpu_device = cudaSetDevice(device_id);

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, device_id);

	// kernel prefers blocks over threads, but does not like only blocks and no threads.
	const int blocks = prop.multiProcessorCount * 2;
	const int numberPerThread = (amount_of_numbers / (blocks * CURAND_NUM_OF_THREADS)) + 1;

	curandState_t *states;
	double *d_nums;

	cudaMalloc(&states, blocks * CURAND_NUM_OF_THREADS * sizeof(curandState_t));
	cudaMalloc(&d_nums, amount_of_numbers * sizeof(double));

	init << <blocks, CURAND_NUM_OF_THREADS >> > (time_seed(), states);

	// this kernel loves bandwidth, so distributing resources should be based on used shared memory.
	// 0 = int = offset (the start of the loop), 1 = int = the end of the loop
	unsigned int sharedMem = (sizeof(int) * 2) + ((sizeof(curandState_t) + sizeof(int)) * CURAND_NUM_OF_THREADS);
	uniform_rand_double_kernel << <blocks, CURAND_NUM_OF_THREADS, sharedMem >> > (states, d_nums, numberPerThread, amount_of_numbers);

	cudaMemcpy(result, d_nums, amount_of_numbers * sizeof(double), cudaMemcpyDeviceToHost);

	cudaFree(states);
	cudaFree(d_nums);
}

extern "C" __declspec(dllexport) void UniformRandDouble(int device_id, int amount_of_numbers, double *result) {
	_uniformRandDouble(device_id, amount_of_numbers, result);
}

__global__ void normal_rand_kernel(curandState_t *states, float *numbers, unsigned int count, unsigned int maximum) {
	// (threadIdx.x * N) + + offset
	// N = number of elements
	// offset = the position of the desired element.
	// 0 = int = offset (the start of the loop), 1 = int = the end of the loop
	extern __shared__ int smem[];

	// unsigned int sharedMem = (sizeof(int) * 2) + ((sizeof(curandState_t) + sizeof(int)) * CURAND_NUM_OF_THREADS);
	int *maximumShared = (int*)&smem[0];
	int *countShared = (int*)&maximumShared[1];
	curandState_t *curandStateShared = (curandState_t*)&countShared[1];
	int *startShared = (int*)&curandStateShared[CURAND_NUM_OF_THREADS];

	if (threadIdx.x == 0) {
		// Make sure we do not go over this.
		maximumShared[0] = maximum;

		// This is the ending point in the *numbers array.
		countShared[0] = count;
	}

	__syncthreads();

	// The state.
	int xid = (threadIdx.x + (blockIdx.x * blockDim.x));
	curandStateShared[threadIdx.x] = states[xid];

	// This is the starting point in the *numbers array.
	startShared[threadIdx.x] = ((threadIdx.x + (blockIdx.x * blockDim.x)) * countShared[0]);

	for (int n = 0; n < countShared[0]; n++) {
		if ((n + startShared[threadIdx.x]) < maximumShared[0]) {
			numbers[(n + startShared[threadIdx.x])] = curand_normal(&curandStateShared[threadIdx.x]);
		}
	}

	states[xid] = curandStateShared[threadIdx.x];
}

void _normalRand(int device_id, int amount_of_numbers, float *result) {
	cudaError_t gpu_device = cudaSetDevice(device_id);

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, device_id);

	// uniform_rand_kernel prefers blocks over threads, but does not like only blocks and no threads.
	const int blocks = prop.maxThreadsDim[0];

	int numberPerThread = (amount_of_numbers / (blocks * CURAND_NUM_OF_THREADS)) + 1;
	// This has to be even, because the kernel generates two numbers per iteration.
	//numberPerThread = numberPerThread - (numberPerThread % 2);

	curandState_t *states;
	float *d_nums;

	cudaMalloc(&states, blocks * CURAND_NUM_OF_THREADS * sizeof(curandState_t));
	cudaMalloc(&d_nums, amount_of_numbers * sizeof(float));

	init << <blocks, CURAND_NUM_OF_THREADS >> > (time_seed(), states);

	// this kernel loves bandwidth, so distributing resources should be based on used shared memory.
	// 0 = int = offset (the start of the loop), 1 = int = the end of the loop
	unsigned int sharedMem = (sizeof(int) * 2) + ((sizeof(curandState_t) + sizeof(int)) * CURAND_NUM_OF_THREADS);
	normal_rand_kernel << <blocks, CURAND_NUM_OF_THREADS, sharedMem >> >(states, d_nums, numberPerThread, amount_of_numbers);

	cudaMemcpy(result, d_nums, amount_of_numbers * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(states);
	cudaFree(d_nums);
}

extern "C" __declspec(dllexport) void NormalRand(int device_id, int amount_of_numbers, float *result) {
	_normalRand(device_id, amount_of_numbers, result);
}

__global__ void normal_rand_double_kernel(curandState_t *states, double *numbers, unsigned int count, unsigned int maximum) {
	// (threadIdx.x * N) + + offset
	// N = number of elements
	// offset = the position of the desired element.
	// 0 = int = offset (the start of the loop), 1 = int = the end of the loop
	extern __shared__ int smem[];

	// unsigned int sharedMem = (sizeof(int) * 2) + ((sizeof(curandState_t) + sizeof(int)) * CURAND_NUM_OF_THREADS);
	int *maximumShared = (int*)&smem[0];
	int *countShared = (int*)&maximumShared[1];
	curandState_t *curandStateShared = (curandState_t*)&countShared[1];
	int *startShared = (int*)&curandStateShared[CURAND_NUM_OF_THREADS];

	if (threadIdx.x == 0) {
		// Make sure we do not go over this.
		maximumShared[0] = maximum;

		// This is the ending point in the *numbers array.
		countShared[0] = count;
	}

	__syncthreads();

	// The state.
	int xid = (threadIdx.x + (blockIdx.x * blockDim.x));
	curandStateShared[threadIdx.x] = states[xid];

	// This is the starting point in the *numbers array.
	startShared[threadIdx.x] = ((threadIdx.x + (blockIdx.x * blockDim.x)) * countShared[0]);

	for (int n = 0; n < countShared[0]; n++) {
		if ((n + startShared[threadIdx.x]) < maximumShared[0]) {
			numbers[(n + startShared[threadIdx.x])] = curand_normal_double(&curandStateShared[threadIdx.x]);
		}
	}

	states[xid] = curandStateShared[threadIdx.x];
}

void _normalRandDouble(int device_id, int amount_of_numbers, double *result) {
	cudaError_t gpu_device = cudaSetDevice(device_id);

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, device_id);

	// uniform_rand_kernel prefers blocks over threads, but does not like only blocks and no threads.
	const int blocks = prop.maxThreadsDim[0];

	int numberPerThread = (amount_of_numbers / (blocks * CURAND_NUM_OF_THREADS)) + 1;
	// This has to be even, because the kernel generates two numbers per iteration.
	//numberPerThread = numberPerThread - (numberPerThread % 2);

	curandState_t *states;
	double *d_nums;

	cudaMalloc(&states, blocks * CURAND_NUM_OF_THREADS * sizeof(curandState_t));
	cudaMalloc(&d_nums, amount_of_numbers * sizeof(double));

	init << <blocks, CURAND_NUM_OF_THREADS >> > (time_seed(), states);

	// this kernel loves bandwidth, so distributing resources should be based on used shared memory.
	// 0 = int = offset (the start of the loop), 1 = int = the end of the loop
	unsigned int sharedMem = (sizeof(int) * 2) + ((sizeof(curandState_t) + sizeof(int)) * CURAND_NUM_OF_THREADS);
	normal_rand_double_kernel << <blocks, CURAND_NUM_OF_THREADS, sharedMem >> >(states, d_nums, numberPerThread, amount_of_numbers);

	cudaMemcpy(result, d_nums, amount_of_numbers * sizeof(double), cudaMemcpyDeviceToHost);

	cudaFree(states);
	cudaFree(d_nums);
}

extern "C" __declspec(dllexport) void NormalRandDouble(int device_id, int amount_of_numbers, double *result) {
	_normalRandDouble(device_id, amount_of_numbers, result);
}

__global__ void log_normal_rand_kernel(curandState_t *states, float *numbers, unsigned int count, unsigned int maximum, float mean, float stddev) {
	// (threadIdx.x * N) + + offset
	// N = number of elements
	// offset = the position of the desired element.
	// 0 = int = offset (the start of the loop), 1 = int = the end of the loop
	extern __shared__ int smem[];

	// unsigned int sharedMem = (sizeof(int) * 2) + (sizeof(float) * 2) + ((sizeof(curandState_t) + sizeof(int)) * CURAND_NUM_OF_THREADS);
	int *maximumShared = (int*)&smem[0];
	int *countShared = (int*)&maximumShared[1];
	float *meanShared = (float*)&countShared[1];
	float *stdDevShared = (float*)&meanShared[1];
	curandState_t *curandStateShared = (curandState_t*)&stdDevShared[1];
	int *startShared = (int*)&curandStateShared[CURAND_NUM_OF_THREADS];

	if (threadIdx.x == 0) {
		// Make sure we do not go over this.
		maximumShared[0] = maximum;

		// This is the ending point in the *numbers array.
		countShared[0] = count;

		// The mean.
		meanShared[0] = mean;

		//The standard deviation.
		stdDevShared[0] = stddev;
	}

	__syncthreads();

	// The state.
	int xid = (threadIdx.x + (blockIdx.x * blockDim.x));
	curandStateShared[threadIdx.x] = states[xid];

	// This is the starting point in the *numbers array.
	startShared[threadIdx.x] = ((threadIdx.x + (blockIdx.x * blockDim.x)) * countShared[0]);

	for (int n = 0; n < countShared[0]; n++) {
		if ((n + startShared[threadIdx.x]) < maximumShared[0]) {
			numbers[(n + startShared[threadIdx.x])] = curand_log_normal(&curandStateShared[threadIdx.x], meanShared[0], stdDevShared[0]);
		}
	}

	states[xid] = curandStateShared[threadIdx.x];
}

void _logNormalRand(int device_id, int amount_of_numbers, float *result, float mean, float stddev) {
	cudaError_t gpu_device = cudaSetDevice(device_id);

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, device_id);

	// uniform_rand_kernel prefers blocks over threads, but does not like only blocks and no threads.
	const int blocks = prop.maxThreadsDim[0];
	const int numberPerThread = (amount_of_numbers / (blocks * CURAND_NUM_OF_THREADS)) + 1;

	curandState_t *states;
	float *d_nums;

	cudaMalloc(&states, blocks * CURAND_NUM_OF_THREADS * sizeof(curandState_t));
	cudaMalloc(&d_nums, amount_of_numbers * sizeof(float));

	init << <blocks, CURAND_NUM_OF_THREADS >> > (time_seed(), states);

	// this kernel loves bandwidth, so distributing resources should be based on used shared memory.
	// 0 = int = offset (the start of the loop), 1 = int = the end of the loop
	unsigned int sharedMem = (sizeof(int) * 2) + (sizeof(float) * 2) + ((sizeof(curandState_t) + sizeof(int)) * CURAND_NUM_OF_THREADS);
	log_normal_rand_kernel << <blocks, CURAND_NUM_OF_THREADS, sharedMem >> >(states, d_nums, numberPerThread, amount_of_numbers, mean, stddev);

	cudaMemcpy(result, d_nums, amount_of_numbers * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(states);
	cudaFree(d_nums);
}

extern "C" __declspec(dllexport) void LogNormalRand(int device_id, int amount_of_numbers, float *result, float mean, float stddev) {
	_logNormalRand(device_id, amount_of_numbers, result, mean, stddev);
}

__global__ void log_normal_rand_double_kernel(curandState_t *states, double *numbers, unsigned int count, unsigned int maximum, float mean, float stddev) {
	// (threadIdx.x * N) + + offset
	// N = number of elements
	// offset = the position of the desired element.
	// 0 = int = offset (the start of the loop), 1 = int = the end of the loop
	extern __shared__ int smem[];

	// unsigned int sharedMem = (sizeof(int) * 2) + (sizeof(float) * 2) + ((sizeof(curandState_t) + sizeof(int)) * CURAND_NUM_OF_THREADS);
	int *maximumShared = (int*)&smem[0];
	int *countShared = (int*)&maximumShared[1];
	float *meanShared = (float*)&countShared[1];
	float *stdDevShared = (float*)&meanShared[1];
	curandState_t *curandStateShared = (curandState_t*)&stdDevShared[1];
	int *startShared = (int*)&curandStateShared[CURAND_NUM_OF_THREADS];

	if (threadIdx.x == 0) {
		// Make sure we do not go over this.
		maximumShared[0] = maximum;

		// This is the ending point in the *numbers array.
		countShared[0] = count;

		// The mean.
		meanShared[0] = mean;

		//The standard deviation.
		stdDevShared[0] = stddev;
	}

	__syncthreads();

	// The state.
	int xid = (threadIdx.x + (blockIdx.x * blockDim.x));
	curandStateShared[threadIdx.x] = states[xid];

	// This is the starting point in the *numbers array.
	startShared[threadIdx.x] = ((threadIdx.x + (blockIdx.x * blockDim.x)) * countShared[0]);

	for (int n = 0; n < countShared[0]; n++) {
		if ((n + startShared[threadIdx.x]) < maximumShared[0]) {
			numbers[(n + startShared[threadIdx.x])] = curand_log_normal_double(&curandStateShared[threadIdx.x], meanShared[0], stdDevShared[0]);
		}
	}

	states[xid] = curandStateShared[threadIdx.x];
}

void _logNormalRandDouble(int device_id, int amount_of_numbers, double *result, float mean, float stddev) {
	cudaError_t gpu_device = cudaSetDevice(device_id);

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, device_id);

	// uniform_rand_kernel prefers blocks over threads, but does not like only blocks and no threads.
	const int blocks = prop.maxThreadsDim[0];
	const int numberPerThread = (amount_of_numbers / (blocks * CURAND_NUM_OF_THREADS)) + 1;

	curandState_t *states;
	double *d_nums;

	cudaMalloc(&states, blocks * CURAND_NUM_OF_THREADS * sizeof(curandState_t));
	cudaMalloc(&d_nums, amount_of_numbers * sizeof(double));

	init << <blocks, CURAND_NUM_OF_THREADS >> > (time_seed(), states);

	// this kernel loves bandwidth, so distributing resources should be based on used shared memory.
	// 0 = int = offset (the start of the loop), 1 = int = the end of the loop
	unsigned int sharedMem = (sizeof(int) * 2) + (sizeof(float) * 2) + ((sizeof(curandState_t) + sizeof(int)) * CURAND_NUM_OF_THREADS);
	log_normal_rand_double_kernel << <blocks, CURAND_NUM_OF_THREADS, sharedMem >> >(states, d_nums, numberPerThread, amount_of_numbers, mean, stddev);

	cudaMemcpy(result, d_nums, amount_of_numbers * sizeof(double), cudaMemcpyDeviceToHost);

	cudaFree(states);
	cudaFree(d_nums);
}

extern "C" __declspec(dllexport) void LogNormalRandDouble(int device_id, int amount_of_numbers, double *result, float mean, float stddev) {
	_logNormalRandDouble(device_id, amount_of_numbers, result, mean, stddev);
}

__global__ void poisson_rand_kernel(curandState_t *states, int *numbers, unsigned int count, unsigned int maximum, double lambda) {
	extern __shared__ int smem[];

	// unsigned int sharedMem = (sizeof(int) * 2) + sizeof(double) + ((sizeof(curandState_t) + sizeof(int) + sizeof(int)) * CURAND_NUM_OF_THREADS);
	int *maximumShared = (int*)&smem[0];
	int *countShared = (int*)&maximumShared[1];
	double *lambdaShared = (double*)&countShared[1];
	curandState_t *curandStateShared = (curandState_t*)&lambdaShared[1];
	int *startShared = (int*)&curandStateShared[CURAND_NUM_OF_THREADS];

	if (threadIdx.x == 0) {
		// Make sure we do not go over this.
		maximumShared[0] = maximum;

		// This is the ending point in the *numbers array.
		countShared[0] = count;

		// The lambda.
		lambdaShared[0] = lambda;
	}

	__syncthreads();

	// The state.
	int xid = (threadIdx.x + (blockIdx.x * blockDim.x));
	curandStateShared[threadIdx.x] = states[xid];

	// This is the starting point in the *numbers array.
	startShared[threadIdx.x] = ((threadIdx.x + (blockIdx.x * blockDim.x)) * countShared[0]);

	for (int n = 0; n < countShared[0]; n++) {
		if ((n + startShared[threadIdx.x]) < maximumShared[0]) {
			numbers[(n + startShared[threadIdx.x])] = curand_poisson(&curandStateShared[threadIdx.x], lambdaShared[0]);
		}
	}

	states[xid] = curandStateShared[threadIdx.x];
}

void _poissonRand(int device_id, int amount_of_numbers, int *result, double lambda) {
	cudaError_t gpu_device = cudaSetDevice(device_id);

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, device_id);

	// kernel prefers blocks over threads, but does not like only blocks and no threads.
	const int blocks = prop.multiProcessorCount * 2;
	const int numberPerThread = (amount_of_numbers / (blocks * CURAND_NUM_OF_THREADS)) + 1;

	curandState_t *states;
	int *d_nums;

	cudaMalloc(&states, blocks * CURAND_NUM_OF_THREADS * sizeof(curandState_t));
	cudaMalloc(&d_nums, amount_of_numbers * sizeof(int));

	init << <blocks, CURAND_NUM_OF_THREADS >> > (time_seed(), states);

	// this kernel loves bandwidth, so distributing resources should be based on used shared memory.
	// 0 = int = offset (the start of the loop), 1 = int = the end of the loop
	unsigned int sharedMem = (sizeof(int) * 2) + sizeof(double) + ((sizeof(curandState_t) + sizeof(int)) * CURAND_NUM_OF_THREADS);
	poisson_rand_kernel << <blocks, CURAND_NUM_OF_THREADS, sharedMem >> > (states, d_nums, numberPerThread, amount_of_numbers, lambda);

	cudaMemcpy(result, d_nums, amount_of_numbers * sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(states);
	cudaFree(d_nums);
}

extern "C" __declspec(dllexport) void PoissonRand(int device_id, int amount_of_numbers, int *result, double lambda) {
	_poissonRand(device_id, amount_of_numbers, result, lambda);
}

