#pragma once
#include <stdint.h>
#include "cuda_blas.h"
#include "cuStats.h"
#include "cuArray.h"
#include "cuda_rand.h"
#include "device_info.h"

// These functions are intended to be used in external sources, such as Platform Invoke in .NET.
// If you plan to program in C++, then cfunc.h and cfunc.cpp can be deleted. These functions are not
// and should not be referenced in the rest of the library.

extern "C" {

	// C does not "support" function overloading like C++ does.
	// Why, then, do these have to be marked as C? C++ will mangle the function names to support overloading.
	// Marking them as C will make sure that these function names will not be changed.
	
	// cuda_blas.h
	__declspec(dllexport) int MatrixMultiplyFloat(
		int32_t device_id,
		int32_t transa_op, int32_t transb_op,
		int32_t m, int32_t n, int32_t k,
		float alpha,
		float *a,
		float *b,
		float beta,
		float *c);
	__declspec(dllexport) int MatrixMultiplyDouble(
		int32_t device_id,
		int32_t transa_op, int32_t transb_op,
		int32_t m, int32_t n, int32_t k,
		double alpha,
		double *a,
		double *b,
		double beta,
		double *c);

	// cuStats.h
	__declspec(dllexport) csl::cuStats* CreateStatClass(int32_t device_id, int64_t amount_of_numbers);
	__declspec(dllexport) void DisposeStatClass(csl::cuStats* stat);

	__declspec(dllexport) int32_t StandardDeviationFloat(csl::cuStats* stat, double &result, float *population, const int64_t population_size, float mean);
	__declspec(dllexport) int32_t StandardDeviationDouble(csl::cuStats* stat, double &result, double *population, const int64_t population_size, double mean);

	__declspec(dllexport) int32_t SampleStandardDeviationFloat(csl::cuStats* stat, double &result, float *sample, const int64_t sample_size, float mean);
	__declspec(dllexport) int32_t SampleStandardDeviationDouble(csl::cuStats* stat, double &result, double *sample, const int64_t sample_size, double mean);

	__declspec(dllexport) int32_t SampleCovarianceFloat(csl::cuStats* stat, double &result, float *x_array, float x_mean, float *y_array, float y_mean, const int64_t array_size);
	__declspec(dllexport) int32_t SampleCovarianceDouble(csl::cuStats* stat, double &result, double *x_array, double x_mean, double *y_array, double y_mean, const int64_t array_size);

	__declspec(dllexport) int32_t CovarianceFloat(csl::cuStats* stat, double &result, float *x_array, float x_mean, float *y_array, float y_mean, const int64_t array_size);
	__declspec(dllexport) int32_t CovarianceDouble(csl::cuStats* stat, double &result, double *x_array, double x_mean, double *y_array, double y_mean, const int64_t array_size);

	__declspec(dllexport) int32_t PearsonCorrelationFloat(csl::cuStats* stat, double &result, float *x_array, float x_mean, float *y_array, float y_mean, const int64_t array_size);
	__declspec(dllexport) int32_t PearsonCorrelationDouble(csl::cuStats* stat, double &result, double *x_array, double x_mean, double *y_array, double y_mean, const int64_t array_size);

	// cuArray.h
	__declspec(dllexport) csl::cuArray* CreateArrayClass(int32_t device_id, int64_t amount_of_numbers);
	__declspec(dllexport) void DisposeArrayClass(csl::cuArray* rand);

	__declspec(dllexport) int32_t AddIntArrays(csl::cuArray* arr, int32_t *result, int32_t *array1, int32_t *array2, const int64_t full_idx);
	__declspec(dllexport) int32_t AddFloatArrays(csl::cuArray* arr, float *result, float *array1, float *array2, const int64_t full_idx);
	__declspec(dllexport) int32_t AddLongArrays(csl::cuArray* arr, int64_t *result, int64_t *array1, int64_t *array2, const int64_t full_idx);
	__declspec(dllexport) int32_t AddDoubleArrays(csl::cuArray* arr, double *result, double *array1, double *array2, const int64_t full_idx);

	__declspec(dllexport) int32_t SubtractIntArrays(csl::cuArray* arr, int32_t *result, int32_t *array1, int32_t *array2, const int64_t full_idx);
	__declspec(dllexport) int32_t SubtractFloatArrays(csl::cuArray* arr, float *result, float *array1, float *array2, const int64_t full_idx);
	__declspec(dllexport) int32_t SubtractLongArrays(csl::cuArray* arr, int64_t *result, int64_t *array1, int64_t *array2, const int64_t full_idx);
	__declspec(dllexport) int32_t SubtractDoubleArrays(csl::cuArray* arr, double *result, double *array1, double *array2, const int64_t full_idx);

	// cuda_rand.h
	__declspec(dllexport) csl::cuda_rand* CreateRandomClass(int32_t device_id, int64_t amount_of_numbers);
	__declspec(dllexport) void DisposeRandomClass(csl::cuda_rand* rand);

	__declspec(dllexport) int32_t UniformRand(csl::cuda_rand* device_id, float *result, int64_t amount_of_numbers);
	__declspec(dllexport) int32_t UniformRandDouble(csl::cuda_rand* device_id, double *result, int64_t amount_of_numbers);
	__declspec(dllexport) int32_t NormalRand(csl::cuda_rand* device_id, float *result, int64_t amount_of_numbers);
	__declspec(dllexport) int32_t NormalRandDouble(csl::cuda_rand* device_id, double *result, int64_t amount_of_numbers);
	__declspec(dllexport) int32_t LogNormalRand(csl::cuda_rand* device_id, float *result, int64_t amount_of_numbers, float mean, float stddev);
	__declspec(dllexport) int32_t LogNormalRandDouble(csl::cuda_rand* device_id, double *result, int64_t amount_of_numbers, float mean, float stddev);
	__declspec(dllexport) int32_t PoissonRand(csl::cuda_rand* device_id, int32_t *result, int64_t amount_of_numbers, double lambda);

	// DeviceInfo.h
	__declspec(dllexport) int32_t GetCudaDeviceCount();
	__declspec(dllexport) int32_t GetCudaDeviceName(int32_t device_id, char* device_name_ptr);
	__declspec(dllexport) int32_t ResetCudaDevice();
}