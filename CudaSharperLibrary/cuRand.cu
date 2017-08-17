#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_profiler_api.h"
#include "curand.h"
#include "curand_kernel.h"

__global__ void init(unsigned int seed, curandState_t* states) {
	int xid = (threadIdx.x + (blockIdx.x * blockDim.x));
	curand_init(
		xid + seed,
		0,
		0,
		&states[xid]
	);
}

__global__ void uniform_rand_kernel(curandState_t *states, float *numbers, int count) {
	int xid = (threadIdx.x + (blockIdx.x * blockDim.x));

	// This is an offset: it determines the starting point of this kernel's place in the array.
	int offset = (threadIdx.x + (blockIdx.x * blockDim.x)) * count;

	for (int n = offset; n < offset + count; n++) {
		numbers[n] = curand_uniform(&states[xid]);
	}
}

void _uniformRand(int device_id, int amount_of_numbers, float *result) {
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

	init << <blocks, threadsPerBlock >> > (time(0), states);
	uniform_rand_kernel << <blocks, threadsPerBlock >> > (states, d_nums, numberPerThread);

	cudaMemcpy(result, d_nums, amount_of_numbers * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(states);
	cudaFree(d_nums);
}

extern "C" __declspec(dllexport) void UniformRand(int device_id, int amount_of_numbers, float *result) {
	_uniformRand(device_id, amount_of_numbers, result);
}

__global__ void normal_rand_kernel(curandState_t *states, float *numbers, int count) {
	int xid = (threadIdx.x + (blockIdx.x * blockDim.x));

	// This is an offset: it determines the starting point of this kernel's place in the array.
	int offset = (threadIdx.x + (blockIdx.x * blockDim.x)) * count;

	for (int n = offset; n < offset + count; n++) {
		numbers[n] = curand_normal(&states[xid]);
	}
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

	init << <blocks, threadsPerBlock >> > (time(0), states);
	normal_rand_kernel << <blocks, threadsPerBlock >> > (states, d_nums, numberPerThread);

	cudaMemcpy(result, d_nums, amount_of_numbers * sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(states);
	cudaFree(d_nums);
}

extern "C" __declspec(dllexport) void NormalRand(int device_id, int amount_of_numbers, float *result) {
	_normalRand(device_id, amount_of_numbers, result);
}

__global__ void poisson_rand_kernel(curandState_t *states, int *numbers, double lambda, int count) {
	int xid = (threadIdx.x + (blockIdx.x * blockDim.x));

	// This is an offset: it determines the starting point of this kernel's place in the array.
	int offset = (threadIdx.x + (blockIdx.x * blockDim.x)) * count;

	for (int n = offset; n < offset + count; n++) {
		numbers[n] = curand_poisson(&states[xid], lambda);
	}
}

void _poissonRand(int device_id, int amount_of_numbers, int *result, double lambda) {
	cudaError_t gpu_device = cudaSetDevice(device_id);

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, device_id);

	// uniform_rand_kernel prefers blocks over threads, but does not like only blocks and no threads.
	const int threadsPerBlock = prop.warpSize * 2;
	const int blocks = *prop.maxThreadsDim;
	const int numberPerThread = (amount_of_numbers / (blocks * threadsPerBlock)) + 1;

	curandState_t *states;
	int *d_nums;

	cudaMalloc(&states, blocks * threadsPerBlock * sizeof(curandState_t));
	cudaMalloc(&d_nums, amount_of_numbers * sizeof(int));

	init << <blocks, threadsPerBlock >> > (time(0), states);
	poisson_rand_kernel << <blocks, threadsPerBlock >> > (states, d_nums, lambda, numberPerThread);

	cudaMemcpy(result, d_nums, amount_of_numbers * sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(states);
	cudaFree(d_nums);
}

extern "C" __declspec(dllexport) void PoissonRand(int device_id, int amount_of_numbers, int *result, double lambda) {
	_poissonRand(device_id, amount_of_numbers, result, lambda);
}
