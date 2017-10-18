#include "cuda_blas.h"

cublasOperation_t cuda_blas_determine_op(int32_t op) {
	cublasOperation_t trans_op = CUBLAS_OP_N;
	switch (op) {
		case 0: trans_op = CUBLAS_OP_N; break;
		case 1: trans_op = CUBLAS_OP_T; break;
		case 2: trans_op = CUBLAS_OP_C; break;
	}
	return trans_op;
}

cudaError_t cuda_blas_matrix_multiply(
	const int32_t device_id,
	const int32_t transa_op, const int32_t transb_op,
	const int32_t m, const int32_t n, const int32_t k,
	const float alpha,
	const float* __restrict a,
	const float* __restrict b,
	const float beta,
	float* __restrict c) {
	cudaError_t errorCode = cudaSetDevice(device_id);
	if (errorCode != cudaSuccess) return errorCode;

	cudaDeviceProp prop;
	errorCode = cudaGetDeviceProperties(&prop, device_id);
	if (errorCode != cudaSuccess) return errorCode;

	// A(m,k)
	int row_a = m;
	// A(m, k)
	int col_a = k;

	// B(k,n)
	int row_b = k;
	// B(k,n)
	int col_b = n;

	// C(m,n)
	int row_c = m;
	//C(m,n)
	int col_c = n;

	cublasOperation_t trans_op_a = cuda_blas_determine_op(transa_op);
	cublasOperation_t trans_op_b = cuda_blas_determine_op(transb_op);

	cublasHandle_t handle;
	cublasCreate(&handle);

	// .NET does not support marshaling nested arrays between C++ and e.g. C#.
	// If you try, you will get the error message, "There is no marshaling support for nested arrays."
	// The solution: this library will only deal with flatten arrays. If you need to use nested arrays,
	// You must create a wrapper that flattens them before passing them to this function and then
	// unflattens them when this function passes a result back.

	float *d_a, *d_b, *d_c;

	size_t size_a = row_a * col_a * sizeof(float);
	size_t size_b = row_b * col_b * sizeof(float);
	size_t size_c = row_c * col_c * sizeof(float);

	// C(m,n) = A(m,k) * B(k,n)
	errorCode = cudaMalloc(&d_a, size_a);
	if (errorCode != cudaSuccess) return errorCode;
	errorCode = cudaMalloc(&d_b, size_b);
	if (errorCode != cudaSuccess) return errorCode;
	errorCode = cudaMalloc(&d_c, size_c);
	if (errorCode != cudaSuccess) return errorCode;

	errorCode = cudaMemcpy(d_a, a, size_a, cudaMemcpyHostToDevice);
	if (errorCode != cudaSuccess) return errorCode;
	errorCode = cudaMemcpy(d_b, b, size_b, cudaMemcpyHostToDevice);
	if (errorCode != cudaSuccess) return errorCode;

	// C(m,n) = A(m,k) * B(k,n)
	// cuBLAS was written with FORTRAN in mind, which uses a column-major notation for matrices.
	// Native C/C++ code (and, indeed, C#) use row-major notation. This effectively transposes the matrices (but doesn't require any data to be moved around).
	// A simple solution is to swap the matrices.
	// See: http://mccormickml.com/2015/08/29/matrix-multiplication-with-cublas-example/
	// And see: http://peterwittek.com/cublas-matrix-c-style.html
	// Finally: https://gist.github.com/peterwittek/6303527
	cublasSgemm(
		handle,
		trans_op_a, trans_op_b,
		col_b, row_a, col_a,
		&alpha,
		d_b, col_b,
		d_a, col_a,
		&beta,
		d_c, col_b);

	errorCode = cudaMemcpy(c, d_c, size_c, cudaMemcpyDeviceToHost);
	if (errorCode != cudaSuccess) return errorCode;

	errorCode = cudaFree(d_a);
	if (errorCode != cudaSuccess) return errorCode;
	errorCode = cudaFree(d_b);
	if (errorCode != cudaSuccess) return errorCode;
	errorCode = cudaFree(d_c);
	if (errorCode != cudaSuccess) return errorCode;

	cublasDestroy(handle);

	return cudaSuccess;
}

cudaError_t cuda_blas_matrix_multiply(
	const int32_t device_id,
	const int32_t transa_op, const int32_t transb_op,
	const int32_t m, const int32_t n, const int32_t k,
	const double alpha,
	const double* __restrict a,
	const double* __restrict b,
	const double beta,
	double* __restrict c) {
	cudaError_t errorCode = cudaSetDevice(device_id);
	if (errorCode != cudaSuccess) return errorCode;

	cudaDeviceProp prop;
	errorCode = cudaGetDeviceProperties(&prop, device_id);
	if (errorCode != cudaSuccess) return errorCode;

	// A(m,k)
	int row_a = m;
	// A(m, k)
	int col_a = k;

	// B(k,n)
	int row_b = k;
	// B(k,n)
	int col_b = n;

	// C(m,n)
	int row_c = m;
	//C(m,n)
	int col_c = n;

	cublasOperation_t trans_op_a = cuda_blas_determine_op(transa_op);
	cublasOperation_t trans_op_b = cuda_blas_determine_op(transb_op);

	cublasHandle_t handle;
	cublasCreate(&handle);

	// .NET does not support marshaling nested arrays between C++ and e.g. C#.
	// If you try, you will get the error message, "There is no marshaling support for nested arrays."
	// The solution: this library will only deal with flatten arrays. If you need to use nested arrays,
	// You must create a wrapper that flattens them before passing them to this function and then
	// unflattens them when this function passes a result back.

	double *d_a, *d_b, *d_c;

	size_t size_a = row_a * col_a * sizeof(double);
	size_t size_b = row_b * col_b * sizeof(double);
	size_t size_c = row_c * col_c * sizeof(double);

	// C(m,n) = A(m,k) * B(k,n)
	errorCode = cudaMalloc(&d_a, size_a);
	if (errorCode != cudaSuccess) return errorCode;
	errorCode = cudaMalloc(&d_b, size_b);
	if (errorCode != cudaSuccess) return errorCode;
	errorCode = cudaMalloc(&d_c, size_c);
	if (errorCode != cudaSuccess) return errorCode;

	errorCode = cudaMemcpy(d_a, a, size_a, cudaMemcpyHostToDevice);
	if (errorCode != cudaSuccess) return errorCode;
	errorCode = cudaMemcpy(d_b, b, size_b, cudaMemcpyHostToDevice);
	if (errorCode != cudaSuccess) return errorCode;

	// C(m,n) = A(m,k) * B(k,n)
	// cuBLAS was written with FORTRAN in mind, which uses a column-major notation for matrices.
	// Native C/C++ code (and, indeed, C#) use row-major notation. This effectively transposes the matrices (but doesn't require any data to be moved around).
	// A simple solution is to swap the matrices.
	// See: http://mccormickml.com/2015/08/29/matrix-multiplication-with-cublas-example/
	// And see: http://peterwittek.com/cublas-matrix-c-style.html
	// Finally: https://gist.github.com/peterwittek/6303527
	cublasDgemm(
		handle,
		trans_op_a, trans_op_b,
		col_b, row_a, col_a,
		&alpha,
		d_b, col_b,
		d_a, col_a,
		&beta,
		d_c, col_b);

	errorCode = cudaMemcpy(c, d_c, size_c, cudaMemcpyDeviceToHost);
	if (errorCode != cudaSuccess) return errorCode;

	errorCode = cudaFree(d_a);
	if (errorCode != cudaSuccess) return errorCode;
	errorCode = cudaFree(d_b);
	if (errorCode != cudaSuccess) return errorCode;
	errorCode = cudaFree(d_c);
	if (errorCode != cudaSuccess) return errorCode;

	cublasDestroy(handle);

	return cudaSuccess;
}

extern "C" {
	__declspec(dllexport) int MatrixMultiplyFloat(
		int32_t device_id,
		int32_t transa_op, int32_t transb_op,
		int32_t m, int32_t n, int32_t k,
		float alpha,
		float *a,
		float *b,
		float beta,
		float *c) {
		return marshal_cuda_error(cuda_blas_matrix_multiply(device_id, transa_op, transb_op, m, n, k, alpha, a, b, beta, c));
	}

	__declspec(dllexport) int MatrixMultiplyDouble(
		int32_t device_id,
		int32_t transa_op, int32_t transb_op,
		int32_t m, int32_t n, int32_t k,
		double alpha,
		double *a,
		double *b,
		double beta,
		double *c) {
		return marshal_cuda_error(cuda_blas_matrix_multiply(device_id, transa_op, transb_op, m, n, k, alpha, a, b, beta, c));
	}
}
