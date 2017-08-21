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

long time_seed() {
	// time(NULL) is not precise enough to produce different sets of random numbers.
	return std::chrono::system_clock::now().time_since_epoch().count();
}

__global__ void init(long seed, curandState_t* states) {
	curand_init(
		(threadIdx.x + (blockIdx.x * blockDim.x)) + seed,
		0,
		0,
		&states[(threadIdx.x + (blockIdx.x * blockDim.x))]
	);
}

__global__ void uniform_rand_kernel(curandState_t *states, float *numbers, unsigned int count, unsigned int maximum, int numbers_of_threads) {
	// (threadIdx.x * N) + + offset
	// N = number of elements
	// offset = the position of the desired element.
	// 0 = int = offset (the start of the loop), 1 = int = the end of the loop
	extern __shared__ int smem[];

	int *maximumShared = (int*)&smem[0];
	int *countShared = (int*)&maximumShared[1];
	curandState_t *curandStateShared = (curandState_t*)&countShared[1];
	int *startShared = (int*)&curandStateShared[numbers_of_threads];
	int *endShared = (int*)&startShared[numbers_of_threads];

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

	// This is the ending point in the *numbers array.
	endShared[threadIdx.x] = startShared[threadIdx.x] + countShared[0];

	for (int n = startShared[threadIdx.x]; n < endShared[threadIdx.x]; n++) {
		if (n < maximumShared[0]) {
			numbers[n] = curand_uniform(&curandStateShared[threadIdx.x]);
		}
	}

	states[xid] = curandStateShared[threadIdx.x];
}

void _uniformRand(int device_id, int amount_of_numbers, float *result) {
	cudaError_t gpu_device = cudaSetDevice(device_id);

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, device_id);

	// kernel prefers blocks over threads, but does not like only blocks and no threads.
	const int threadsPerBlock = prop.warpSize;
	const int blocks = prop.multiProcessorCount * 2;
	const int numberPerThread = (amount_of_numbers / (blocks * threadsPerBlock)) + 1;

	curandState_t *states;
	float *d_nums;

	cudaMalloc(&states, blocks * threadsPerBlock * sizeof(curandState_t));
	cudaMalloc(&d_nums, amount_of_numbers * sizeof(float));

	init << <blocks, threadsPerBlock >> > (time_seed(), states);

	// this kernel loves bandwidth, so distributing resources should be based on used shared memory.
	// 0 = int = offset (the start of the loop), 1 = int = the end of the loop
	unsigned int sharedMem = (sizeof(int) * 2) + sizeof(double) + ((sizeof(curandState_t) + sizeof(int) + sizeof(int)) * threadsPerBlock);
	uniform_rand_kernel << <blocks, threadsPerBlock, sharedMem >> > (states, d_nums, numberPerThread, amount_of_numbers, threadsPerBlock);

	cudaMemcpy(result, d_nums, amount_of_numbers * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(states);
	cudaFree(d_nums);
}

extern "C" __declspec(dllexport) void UniformRand(int device_id, int amount_of_numbers, float *result) {
	_uniformRand(device_id, amount_of_numbers, result);
}

__global__ void normal_rand_kernel(curandState_t *states, float *numbers, unsigned int count, unsigned int maximum, int numbers_of_threads) {
	// (threadIdx.x * N) + + offset
	// N = number of elements
	// offset = the position of the desired element.
	// 0 = int = offset (the start of the loop), 1 = int = the end of the loop
	extern __shared__ int smem[];

	int *maximumShared = (int*)&smem[0];
	int *countShared = (int*)&maximumShared[1];
	curandState_t *curandStateShared = (curandState_t*)&countShared[1];
	int *startShared = (int*)&curandStateShared[numbers_of_threads];
	int *endShared = (int*)&startShared[numbers_of_threads];

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

	// This is the ending point in the *numbers array.
	endShared[threadIdx.x] = startShared[threadIdx.x] + countShared[0];

	for (int n = startShared[threadIdx.x]; n < endShared[threadIdx.x]; n++) {
		if (n < maximumShared[0]) {
			numbers[n] = curand_normal(&curandStateShared[threadIdx.x]);
		}
	}

	states[xid] = curandStateShared[threadIdx.x];
}

void _normalRand(int device_id, int amount_of_numbers, float *result) {
	cudaError_t gpu_device = cudaSetDevice(device_id);

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, device_id);

	// uniform_rand_kernel prefers blocks over threads, but does not like only blocks and no threads.
	const int threadsPerBlock = prop.warpSize * 2;
	const int blocks = *prop.maxThreadsDim;
	const int numberPerThread = (amount_of_numbers / (blocks * threadsPerBlock)) + 1;

	curandState_t *states;
	float *d_nums;

	cudaMalloc(&states, blocks * threadsPerBlock * sizeof(curandState_t));
	cudaMalloc(&d_nums, amount_of_numbers * sizeof(float));

	init << <blocks, threadsPerBlock >> > (time_seed(), states);

	// this kernel loves bandwidth, so distributing resources should be based on used shared memory.
	// 0 = int = offset (the start of the loop), 1 = int = the end of the loop
	unsigned int sharedMem = (sizeof(int) * 2) + sizeof(double) + ((sizeof(curandState_t) + sizeof(int) + sizeof(int)) * threadsPerBlock);
	normal_rand_kernel<<<blocks, threadsPerBlock, sharedMem>>>(states, d_nums, numberPerThread, amount_of_numbers, threadsPerBlock);

	cudaMemcpy(result, d_nums, amount_of_numbers * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(states);
	cudaFree(d_nums);
}

extern "C" __declspec(dllexport) void NormalRand(int device_id, int amount_of_numbers, float *result) {
	_normalRand(device_id, amount_of_numbers, result);
}

__global__ void poisson_rand_kernel(curandState_t *states, int *numbers, unsigned int count, unsigned int maximum, double lambda, int numbers_of_threads) {
	extern __shared__ int smem[];

	int *maximumShared = (int*)&smem[0];
	int *countShared = (int*)&maximumShared[1];
	double *lambdaShared = (double*)&countShared[1];
	curandState_t *curandStateShared = (curandState_t*)&lambdaShared[1];
	int *startShared = (int*)&curandStateShared[numbers_of_threads];
	int *endShared = (int*)&startShared[numbers_of_threads];

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

	// This is the ending point in the *numbers array.
	endShared[threadIdx.x] = startShared[threadIdx.x] + countShared[0];

	for (int n = startShared[threadIdx.x]; n < endShared[threadIdx.x]; n++) {
		if (n < maximumShared[0]) {
			numbers[n] = curand_poisson(&curandStateShared[threadIdx.x], lambdaShared[0]);
		}
	}

	states[xid] = curandStateShared[threadIdx.x];
}

void _poissonRand(int device_id, int amount_of_numbers, int *result, double lambda) {
	cudaError_t gpu_device = cudaSetDevice(device_id);

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, device_id);

	// kernel prefers blocks over threads, but does not like only blocks and no threads.
	const int threadsPerBlock = prop.warpSize;
	const int blocks = prop.multiProcessorCount * 2;
	const int numberPerThread = (amount_of_numbers / (blocks * threadsPerBlock)) + 1;

	curandState_t *states;
	int *d_nums;

	cudaMalloc(&states, blocks * threadsPerBlock * sizeof(curandState_t));
	cudaMalloc(&d_nums, amount_of_numbers * sizeof(int));

	init << <blocks, threadsPerBlock >> > (time_seed(), states);

	// this kernel loves bandwidth, so distributing resources should be based on used shared memory.
	// 0 = int = offset (the start of the loop), 1 = int = the end of the loop
	unsigned int sharedMem = (sizeof(int) * 2) + sizeof(double) + ((sizeof(curandState_t) + sizeof(int) + sizeof(int)) * threadsPerBlock);
	poisson_rand_kernel << <blocks, threadsPerBlock, sharedMem >> > (states, d_nums, numberPerThread, amount_of_numbers, lambda, threadsPerBlock);

	cudaMemcpy(result, d_nums, amount_of_numbers * sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(states);
	cudaFree(d_nums);
}

extern "C" __declspec(dllexport) void PoissonRand(int device_id, int amount_of_numbers, int *result, double lambda) {
	_poissonRand(device_id, amount_of_numbers, result, lambda);
}
