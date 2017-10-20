using System;
using System.Collections.Generic;

/*
 * CudaSharper - a wrapper for CUDA-accelerated functions. CudaSharper is not intended to write CUDA in C#, but rather a
 * library that allows one to easily use CUDA-accelerated functions without having to directly interact with the device.
 * This file acts as a wrapper for CudaSharperLibrary.dll, which is required for these functions to run. CudaSharperLibrary.dll
 * is the actual CUDA C code compiled as a C++/CLI assembly; however, it is unmanaged and therefore requires this wrapper to be used
 * in C# projects.
 */

namespace CudaSharper
{
    public enum CUBLAS_OP
    {
        DO_NOT_TRANSPOSE = 0,
        TRANSPOSE = 1,
        CONJUGATE = 2
    }

    public sealed class CuArray : IDisposable
    {
        private ICudaDevice CudaDeviceComponent { get; }
        private IntPtr PtrToUnmanagedClass { get; set; }

        public int DeviceId => CudaDeviceComponent.DeviceId;

        static CuArray()
        {
            CudaSettings.Load();
        }

        public CuArray(CudaDevice device)
        {
            CudaDeviceComponent = new CudaDevice(device.DeviceId, device.AllocationSize);
            PtrToUnmanagedClass = SafeNativeMethods.CreateArrayClass(CudaDeviceComponent.DeviceId, CudaDeviceComponent.AllocationSize);
        }

        public ICudaResult<int[]> Add(int[] array1, int[] array2)
        {
            if (array1.Length != array2.Length)
                throw new ArgumentOutOfRangeException("Bad arrays given; they need to be the same length.");

            var result = new int[array1.Length];
            var error = SafeNativeMethods.AddIntArrays(PtrToUnmanagedClass, result, array1, array2, array1.Length);
            return new CudaResult<int[]>(error, result);
        }

        public ICudaResult<float[]> Add(float[] array1, float[] array2)
        {
            if (array1.Length != array2.Length)
                throw new ArgumentOutOfRangeException("Bad arrays given; they need to be the same length.");

            var result = new float[array1.Length];
            var error = SafeNativeMethods.AddFloatArrays(PtrToUnmanagedClass, result, array1, array2, array1.Length);
            return new CudaResult<float[]>(error, result);
        }

        public ICudaResult<long[]> Add(long[] array1, long[] array2)
        {
            if (array1.Length != array2.Length)
                throw new ArgumentOutOfRangeException("Bad arrays given; they need to be the same length.");

            var result = new long[array1.Length];
            var error = SafeNativeMethods.AddLongArrays(PtrToUnmanagedClass, result, array1, array2, array1.Length);
            return new CudaResult<long[]>(error, result);
        }

        public ICudaResult<double[]> Add(double[] array1, double[] array2)
        {
            if (array1.Length != array2.Length)
                throw new ArgumentOutOfRangeException("Bad arrays given; they need to be the same length.");

            var result = new double[array1.Length];
            var error = SafeNativeMethods.AddDoubleArrays(PtrToUnmanagedClass, result, array1, array2, array1.Length);
            return new CudaResult<double[]>(error, result);
        }

        public ICudaResult<int[]> Subtract(int[] array1, int[] array2)
        {
            if (array1.Length != array2.Length)
                throw new ArgumentOutOfRangeException("Bad arrays given; they need to be the same length.");

            var result = new int[array1.Length];
            var error = SafeNativeMethods.SubtractIntArrays(PtrToUnmanagedClass, result, array1, array2, array1.Length);
            return new CudaResult<int[]>(error, result);
        }

        public ICudaResult<float[]> Subtract(float[] array1, float[] array2)
        {
            if (array1.Length != array2.Length)
                throw new ArgumentOutOfRangeException("Bad arrays given; they need to be the same length.");

            var result = new float[array1.Length];
            var error = SafeNativeMethods.SubtractFloatArrays(PtrToUnmanagedClass, result, array1, array2, array1.Length);
            return new CudaResult<float[]>(error, result);
        }

        public ICudaResult<long[]> Subtract(long[] array1, long[] array2)
        {
            if (array1.Length != array2.Length)
                throw new ArgumentOutOfRangeException("Bad arrays given; they need to be the same length.");

            var result = new long[array1.Length];
            var error = SafeNativeMethods.SubtractLongArrays(PtrToUnmanagedClass, result, array1, array2, array1.Length);
            return new CudaResult<long[]>(error, result);
        }

        public ICudaResult<double[]> Subtract(double[] array1, double[] array2)
        {
            if (array1.Length != array2.Length)
                throw new ArgumentOutOfRangeException("Bad arrays given; they need to be the same length.");

            var result = new double[array1.Length];
            var error = SafeNativeMethods.SubtractDoubleArrays(PtrToUnmanagedClass, result, array1, array2, array1.Length);
            return new CudaResult<double[]>(error, result);
        }

        private (int Rows, int Columns) MatrixSizeByOperation(int rows, int columns, CUBLAS_OP operation)
        {
            int matrix_rows = 0;
            int matric_columns = 0;

            switch (operation)
            {
                case CUBLAS_OP.DO_NOT_TRANSPOSE:
                    matrix_rows = rows;
                    matric_columns = columns;
                    break;
                case CUBLAS_OP.TRANSPOSE:
                    matrix_rows = columns;
                    matric_columns = rows;
                    break;
            }

            return (matrix_rows, matric_columns);
        }

        public ICudaResult<float[][]> Multiply(
            CUBLAS_OP a_op,
            CUBLAS_OP b_op,
            float alpha,
            float[][] a,
            float[][] b,
            float beta)
        {
            // C(m, n) = A(m, k) * B(k, n)
            var matrix_a_dimensions = MatrixSizeByOperation(a.Length, a[0].Length, a_op);
            var matrix_b_dimensions = MatrixSizeByOperation(b.Length, b[0].Length, b_op);

            if (matrix_a_dimensions.Columns != matrix_b_dimensions.Rows)
                throw new ArgumentOutOfRangeException($"Matrices provided cannot be multipled. Columns in matrix A: {matrix_a_dimensions.Columns} vs rows in matrix B: {matrix_b_dimensions.Rows}");

            var result = DTM.MatrixMultiplyFloat(
                DeviceId,
                a_op, b_op,
                alpha,
                a,
                b,
                beta);

            return new CudaResult<float[][]>(result.Error, result.Result);
        }

        public ICudaResult<double[][]> Multiply(
            CUBLAS_OP a_op,
            CUBLAS_OP b_op,
            double alpha,
            double[][] a,
            double[][] b,
            double beta)
        {
            // C(m, n) = A(m, k) * B(k, n)
            var matrix_a_dimensions = MatrixSizeByOperation(a.Length, a[0].Length, a_op);
            var matrix_b_dimensions = MatrixSizeByOperation(b.Length, b[0].Length, b_op);

            if (matrix_a_dimensions.Columns != matrix_b_dimensions.Rows)
                throw new ArgumentOutOfRangeException($"Matrices provided cannot be multipled. Columns in matrix A: {matrix_a_dimensions.Columns} vs rows in matrix B: {matrix_b_dimensions.Rows}");

            var result = DTM.MatrixMultiplyDouble(
                DeviceId,
                a_op, b_op,
                alpha,
                a,
                b,
                beta);

            return new CudaResult<double[][]>(result.Error, result.Result);
        }

        #region IDisposable Support
        private bool disposedValue = false; // To detect redundant calls

        void Dispose(bool disposing)
        {
            if (!disposedValue)
            {
                if (disposing)
                {
                    // TODO: dispose managed state (managed objects).
                }

                // TODO: free unmanaged resources (unmanaged objects) and override a finalizer below.
                // TODO: set large fields to null.
                if (PtrToUnmanagedClass != IntPtr.Zero)
                {
                    SafeNativeMethods.DisposeArrayClass(PtrToUnmanagedClass);
                    PtrToUnmanagedClass = IntPtr.Zero;
                }

                disposedValue = true;
            }
        }

        // TODO: override a finalizer only if Dispose(bool disposing) above has code to free unmanaged resources.
        ~CuArray()
        {
            // Do not change this code. Put cleanup code in Dispose(bool disposing) above.
            Dispose(false);
        }

        // This code added to correctly implement the disposable pattern.
        public void Dispose()
        {
            // Do not change this code. Put cleanup code in Dispose(bool disposing) above.
            Dispose(true);
            // TODO: uncomment the following line if the finalizer is overridden above.
            GC.SuppressFinalize(this);
        }
        #endregion
    }
}
