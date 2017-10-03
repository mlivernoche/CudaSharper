#pragma once
#pragma comment(lib,"cublas.lib")
#include "cuda.h"
#include "cuda_runtime.h"
#include "cublas_v2.h"
#include "device_launch_parameters.h"
#include "cuda_profiler_api.h"
#include "device_functions.h"
#include <iostream>

#define IDX2C(i,j,ld) (((j)*(ld))+(i))

cublasOperation_t cuda_blas_determine_op(unsigned int op);

void cuda_blas_matrix_multiply(
	unsigned int device_id,
	unsigned int transa_op, unsigned int transb_op,
	int m, int n, int k,
	float alpha,
	float *a,
	float *b,
	float beta,
	float *c);
void cuda_blas_matrix_multiply(
	unsigned int device_id,
	unsigned int transa_op, unsigned int transb_op,
	int m, int n, int k,
	double alpha,
	double *a,
	double *b,
	double beta,
	double *c);
extern "C" __declspec(dllexport) void MatrixMultiplyFloat(
	unsigned int device_id,
	unsigned int transa_op, unsigned int transb_op,
	int m, int n, int k,
	float alpha,
	float *a,
	float *b,
	float beta,
	float *c);
extern "C" __declspec(dllexport) void MatrixMultiplyDouble(
	unsigned int device_id,
	unsigned int transa_op, unsigned int transb_op,
	int m, int n, int k,
	double alpha,
	double *a,
	double *b,
	double beta,
	double *c);