#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_profiler_api.h"
#include "curand.h"
#include "curand_kernel.h"
#include <iostream>
#include <algorithm>
#include <time.h>

long time_seed();

__global__ void init(long seed, curandState_t* states);

__global__ void uniform_rand_kernel(curandState_t *states, float *numbers, unsigned int count, unsigned int maximum);
void _uniformRand(int device_id, int amount_of_numbers, float *result);
extern "C" __declspec(dllexport) void UniformRand(int device_id, int amount_of_numbers, float *result);

__global__ void normal_rand_kernel(curandState_t *states, float *numbers, unsigned int count, unsigned int maximum);
void _normalRand(int device_id, int amount_of_numbers, float *result);
extern "C" __declspec(dllexport) void NormalRand(int device_id, int amount_of_numbers, float *result);

__global__ void log_normal_rand_kernel(curandState_t *states, float *numbers, unsigned int count, unsigned int maximum, float mean, float stddev);
void _logNormalRand(int device_id, int amount_of_numbers, float *result, float mean, float stddev);
extern "C" __declspec(dllexport) void LogNormalRand(int device_id, int amount_of_numbers, float *result, float mean, float stddev);

__global__ void poisson_rand_kernel(curandState_t *states, int *numbers, unsigned int count, unsigned int maximum, double lambda);
void _poissonRand(int device_id, int amount_of_numbers, int *result, double lambda);
extern "C" __declspec(dllexport) void PoissonRand(int device_id, int amount_of_numbers, int *result, double lambda);
