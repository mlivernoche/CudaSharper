#pragma once
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_profiler_api.h"
#include "curand.h"
#include "curand_kernel.h"
#include <iostream>
#include <algorithm>
#include <time.h>
#include <chrono>

long long int time_seed();

__global__ void init(long long int seed, curandState_t* states);

__global__ void uniform_rand_kernel(curandState_t *states, float *numbers, unsigned int count, unsigned int maximum);
void _uniformRand(unsigned int device_id, unsigned int amount_of_numbers, float *result);
extern "C" __declspec(dllexport) void UniformRand(unsigned int device_id, unsigned int amount_of_numbers, float *result) {
	_uniformRand(device_id, amount_of_numbers, result);
}

__global__ void uniform_rand_double_kernel(curandState_t *states, double *numbers, unsigned int count, unsigned int maximum);
void _uniformRandDouble(unsigned int device_id, unsigned int amount_of_numbers, double *result);
extern "C" __declspec(dllexport) void UniformRandDouble(unsigned int device_id, unsigned int amount_of_numbers, double *result) {
	_uniformRandDouble(device_id, amount_of_numbers, result);
}

__global__ void normal_rand_kernel(curandState_t *states, float *numbers, unsigned int count, unsigned int maximum);
void _normalRand(unsigned int device_id, unsigned int amount_of_numbers, float *result);
extern "C" __declspec(dllexport) void NormalRand(unsigned int device_id, unsigned int amount_of_numbers, float *result) {
	_normalRand(device_id, amount_of_numbers, result);
}

__global__ void normal_rand_double_kernel(curandState_t *states, double *numbers, unsigned int count, unsigned int maximum);
void _normalRandDouble(unsigned int device_id, unsigned int amount_of_numbers, double *result);
extern "C" __declspec(dllexport) void NormalRandDouble(unsigned int device_id, unsigned int amount_of_numbers, double *result) {
	_normalRandDouble(device_id, amount_of_numbers, result);
}

__global__ void log_normal_rand_kernel(curandState_t *states, float *numbers, unsigned int count, unsigned int maximum, float mean, float stddev);
void _logNormalRand(unsigned int device_id, unsigned int amount_of_numbers, float *result, float mean, float stddev);
extern "C" __declspec(dllexport) void LogNormalRand(unsigned int device_id, unsigned int amount_of_numbers, float *result, float mean, float stddev) {
	_logNormalRand(device_id, amount_of_numbers, result, mean, stddev);
}

__global__ void log_normal_rand_double_kernel(curandState_t *states, double *numbers, unsigned int count, unsigned int maximum, float mean, float stddev);
void _logNormalRandDouble(unsigned int device_id, unsigned int amount_of_numbers, double *result, float mean, float stddev);
extern "C" __declspec(dllexport) void LogNormalRandDouble(unsigned int device_id, unsigned int amount_of_numbers, double *result, float mean, float stddev) {
	_logNormalRandDouble(device_id, amount_of_numbers, result, mean, stddev);
}

__global__ void poisson_rand_kernel(curandState_t *states, int *numbers, unsigned int count, unsigned int maximum, double lambda);
void _poissonRand(unsigned int device_id, unsigned int amount_of_numbers, int *result, double lambda);
extern "C" __declspec(dllexport) void PoissonRand(unsigned int device_id, unsigned int amount_of_numbers, int *result, double lambda) {
	_poissonRand(device_id, amount_of_numbers, result, lambda);
}
