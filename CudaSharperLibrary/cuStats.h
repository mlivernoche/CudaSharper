#pragma once
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_profiler_api.h"
#include "device_functions.h"

void cuArray_determine_launch_parameters(unsigned long int* blocks, unsigned long int* threads, unsigned long int* number_per_thread, unsigned long int max_block_size, unsigned long int max_thread_size);

__global__ void cuStats_standard_deviation_kernel(float *std, float *sample, unsigned long int number_per_thread, unsigned long long int data_set_size, double mean);
__global__ void cuStats_standard_deviation_kernel(double *std, double *sample, unsigned long int number_per_thread, unsigned long long int data_set_size, double mean);
template<typename T> double cuStats_standard_deviation(unsigned int device_id, T *sample, unsigned long long int sample_size, double mean);

// C does not "support" function overloading like C++ does.
// Why, then, do these have to be marked as C? C++ will mangle the function names to support overloading.
// Marking them as C will make sure that these function names will not be changed.
// C... simply designed, simply frustrating.
extern "C" __declspec(dllexport) double StandardDeviationFloat(unsigned int device_id, float *population, unsigned long long int population_size, double mean);
extern "C" __declspec(dllexport) double StandardDeviationDouble(unsigned int device_id, double *population, unsigned long long int population_size, double mean);

template<typename T> double cuStats_sample_standard_deviation(unsigned int device_id, T *sample, unsigned long long int sample_size, double mean);

// C does not "support" function overloading like C++ does.
// Why, then, do these have to be marked as C? C++ will mangle the function names to support overloading.
// Marking them as C will make sure that these function names will not be changed.
// C... simply designed, simply frustrating.
extern "C" __declspec(dllexport) double SampleStandardDeviationFloat(unsigned int device_id, float *sample, unsigned long long int sample_size, double mean);
extern "C" __declspec(dllexport) double SampleStandardDeviationDouble(unsigned int device_id, double *sample, unsigned long long int sample_size, double mean);
