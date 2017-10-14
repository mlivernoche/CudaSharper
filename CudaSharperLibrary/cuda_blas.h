#pragma once
#pragma comment(lib,"cublas.lib")
#include "cuda.h"
#include "cuda_runtime.h"
#include "cublas_v2.h"
#include "device_launch_parameters.h"
#include "cuda_profiler_api.h"
#include "device_functions.h"
#include <iostream>

#include "DeviceInfo.h"

cublasOperation_t cuda_blas_determine_op(int32_t op);

cudaError_t cuda_blas_matrix_multiply(
	int32_t device_id,
	int32_t transa_op, int32_t transb_op,
	int32_t m, int32_t n, int32_t k,
	float alpha,
	float *a,
	float *b,
	float beta,
	float *c);
cudaError_t cuda_blas_matrix_multiply(
	int32_t device_id,
	int32_t transa_op, int32_t transb_op,
	int32_t m, int32_t n, int32_t k,
	double alpha,
	double *a,
	double *b,
	double beta,
	double *c);
extern "C" {
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
}