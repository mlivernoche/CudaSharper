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

void cuda_rand_determine_launch_parameters(unsigned long int* blocks, unsigned long int* threads, unsigned long int* number_per_thread, unsigned long int max_block_size, unsigned long int max_thread_size);
long long int cuda_rand_time_seed();

extern "C" __declspec(dllexport) void UniformRand(unsigned int device_id, unsigned int amount_of_numbers, float *result);
extern "C" __declspec(dllexport) void UniformRandDouble(unsigned int device_id, unsigned int amount_of_numbers, double *result);
extern "C" __declspec(dllexport) void NormalRand(unsigned int device_id, unsigned int amount_of_numbers, float *result);
extern "C" __declspec(dllexport) void NormalRandDouble(unsigned int device_id, unsigned int amount_of_numbers, double *result);
extern "C" __declspec(dllexport) void LogNormalRand(unsigned int device_id, unsigned int amount_of_numbers, float *result, float mean, float stddev);
extern "C" __declspec(dllexport) void LogNormalRandDouble(unsigned int device_id, unsigned int amount_of_numbers, double *result, float mean, float stddev);
extern "C" __declspec(dllexport) void PoissonRand(unsigned int device_id, unsigned int amount_of_numbers, int *result, double lambda);

__global__ void cuda_rand_uniform_rand_kernel(long long int seed, float *numbers, unsigned int count, unsigned int maximum);
void cuda_rand_uniform_rand(unsigned int device_id, unsigned int amount_of_numbers, float *result);

__global__ void cuda_rand_uniform_rand_double_kernel(long long int seed, double *numbers, unsigned int count, unsigned int maximum);
void cuda_rand_uniform_rand_double(unsigned int device_id, unsigned int amount_of_numbers, double *result);

__global__ void cuda_rand_normal_rand_kernel(long long int seed, float *numbers, unsigned int count, unsigned int maximum);
void cuda_rand_normal_rand(unsigned int device_id, unsigned int amount_of_numbers, float *result);

__global__ void cuda_rand_normal_rand_double_kernel(long long int seed, double *numbers, unsigned int count, unsigned int maximum);
void cuda_rand_normal_rand_double(unsigned int device_id, unsigned int amount_of_numbers, double *result);

__global__ void cuda_rand_log_normal_rand_kernel(long long int seed, float *numbers, unsigned int count, unsigned int maximum, float mean, float stddev);
void cuda_rand_log_normal_rand(unsigned int device_id, unsigned int amount_of_numbers, float *result, float mean, float stddev);

__global__ void cuda_rand_log_normal_rand_double_kernel(long long int seed, double *numbers, unsigned int count, unsigned int maximum, double mean, double stddev);
void cuda_rand_log_normal_rand_double(unsigned int device_id, unsigned int amount_of_numbers, double *result, double mean, double stddev);

__global__ void cuda_rand_poisson_rand_kernel(long long int seed, int *numbers, unsigned int count, unsigned int maximum, double lambda);
void cuda_rand_poisson_rand(unsigned int device_id, unsigned int amount_of_numbers, int *result, double lambda);