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

    public class CuArray : ICudaDevice
    {
        private ICudaDevice CudaDeviceComponent { get; }

        public int DeviceId => CudaDeviceComponent.DeviceId;

        static CuArray()
        {
            CudaSettings.Load();
        }

        public CuArray(int device_id)
        {
            CudaDeviceComponent = new CudaDevice(device_id);
        }

        public CuArray(CudaDevice device)
        {
            CudaDeviceComponent = device;
        }

        public string GetCudaDeviceName()
        {
            return CudaDeviceComponent.GetCudaDeviceName();
        }

        public void Split(int[] src, int[] array1, int[] array2, uint length, uint split_index)
        {
            if (split_index >= length)
                throw new ArgumentOutOfRangeException(string.Format("Split index is greater then the length of the array. Supplied values: length={0}, split_index={1}.", length, split_index));

            SafeNativeMethods.SplitIntArray(DeviceId, src, array1, array2, length, split_index);
        }

        public (int[] Left, int[] Right) Split(int[] src, uint length, uint split_index)
        {
            if (split_index >= length)
                throw new ArgumentOutOfRangeException(string.Format("Split index is greater then the length of the array. Supplied values: length={0}, split_index={1}.", length, split_index));

            var array1 = new int[split_index];
            var array2 = new int[length - split_index];
            SafeNativeMethods.SplitIntArray(DeviceId, src, array1, array2, length, split_index);
            return (array1, array2);
        }

        public void Split(long[] src, long[] array1, long[] array2, uint length, uint split_index)
        {
            if (split_index >= length)
                throw new ArgumentOutOfRangeException(string.Format("Split index is greater then the length of the array. Supplied values: length={0}, split_index={1}.", length, split_index));

            SafeNativeMethods.SplitLongArray(DeviceId, src, array1, array2, length, split_index);
        }

        public (long[] Left, long[] Right) Split(long[] src, uint length, uint split_index)
        {
            if (split_index >= length)
                throw new ArgumentOutOfRangeException(string.Format("Split index is greater then the length of the array. Supplied values: length={0}, split_index={1}.", length, split_index));

            var array1 = new long[split_index];
            var array2 = new long[length - split_index];
            SafeNativeMethods.SplitLongArray(DeviceId, src, array1, array2, length, split_index);
            return (array1, array2);
        }

        public void Split(float[] src, float[] array1, float[] array2, uint length, uint split_index)
        {
            if (split_index >= length)
                throw new ArgumentOutOfRangeException(string.Format("Split index is greater then the length of the array. Supplied values: length={0}, split_index={1}.", length, split_index));

            SafeNativeMethods.SplitFloatArray(DeviceId, src, array1, array2, length, split_index);
        }

        public (float[] Left, float[] Right) Split(float[] src, uint length, uint split_index)
        {
            if (split_index >= length)
                throw new ArgumentOutOfRangeException(string.Format("Split index is greater then the length of the array. Supplied values: length={0}, split_index={1}.", length, split_index));

            var array1 = new float[split_index];
            var array2 = new float[length - split_index];
            SafeNativeMethods.SplitFloatArray(DeviceId, src, array1, array2, length, split_index);
            return (array1, array2);
        }

        public void Split(double[] src, double[] array1, double[] array2, uint length, uint split_index)
        {
            if (split_index >= length)
                throw new ArgumentOutOfRangeException(string.Format("Split index is greater then the length of the array. Supplied values: length={0}, split_index={1}.", length, split_index));

            SafeNativeMethods.SplitDoubleArray(DeviceId, src, array1, array2, length, split_index);
        }

        public (double[] Left, double[] Right) Split(double[] src, uint length, uint split_index)
        {
            if (split_index >= length)
                throw new ArgumentOutOfRangeException(string.Format("Split index is greater then the length of the array. Supplied values: length={0}, split_index={1}.", length, split_index));

            var array1 = new double[split_index];
            var array2 = new double[length - split_index];
            SafeNativeMethods.SplitDoubleArray(DeviceId, src, array1, array2, length, split_index);
            return (array1, array2);
        }

        public void Merge(int[] result, int[] array1, int[] array2)
        {
            if (array1.Length + array2.Length > Int32.MaxValue)
                throw new Exception("Resultant array is too large.");

            SafeNativeMethods.MergeIntArrays(DeviceId, result, array1, array2, (uint)array1.Length, (uint)array2.Length);
        }

        public int[] Merge(int[] array1, int[] array2)
        {
            if (array1.Length + array2.Length > Int32.MaxValue)
                throw new Exception("Resultant array is too large.");

            var result = new int[array1.Length + array2.Length];
            SafeNativeMethods.MergeIntArrays(DeviceId, result, array1, array2, (uint)array1.Length, (uint)array2.Length);
            return result;
        }

        public void Merge(long[] result, long[] array1, long[] array2)
        {
            if (array1.Length + array2.Length > Int32.MaxValue)
                throw new Exception("Resultant array is too large.");

            SafeNativeMethods.MergeLongArrays(DeviceId, result, array1, array2, (uint)array1.Length, (uint)array2.Length);
        }

        public long[] Merge(long[] array1, long[] array2)
        {
            if (array1.Length + array2.Length > Int32.MaxValue)
                throw new Exception("Resultant array is too large.");

            var result = new long[array1.Length + array2.Length];
            SafeNativeMethods.MergeLongArrays(DeviceId, result, array1, array2, (uint)array1.Length, (uint)array2.Length);
            return result;
        }

        public void Merge(float[] result, float[] array1, float[] array2)
        {
            if (array1.Length + array2.Length > Int32.MaxValue)
                throw new Exception("Resultant array is too large.");

            SafeNativeMethods.MergeFloatArrays(DeviceId, result, array1, array2, array1.Length, array2.Length);
        }

        public float[] Merge(float[] array1, float[] array2)
        {
            if (array1.Length + array2.Length > Int32.MaxValue)
                throw new Exception("Resultant array is too large.");

            var result = new float[array1.Length + array2.Length];
            SafeNativeMethods.MergeFloatArrays(DeviceId, result, array1, array2, array1.Length, array2.Length);
            return result;
        }

        public void Merge(double[] result, double[] array1, double[] array2)
        {
            if (array1.Length + array2.Length > Int32.MaxValue)
                throw new Exception("Resultant array is too large.");

            SafeNativeMethods.MergeDoubleArrays(DeviceId, result, array1, array2, array1.Length, array2.Length);
        }

        public double[] Merge(double[] array1, double[] array2)
        {
            if (array1.Length + array2.Length > Int32.MaxValue)
                throw new Exception("Resultant array is too large.");

            var result = new double[array1.Length + array2.Length];
            SafeNativeMethods.MergeDoubleArrays(DeviceId, result, array1, array2, array1.Length, array2.Length);
            return result;
        }

        public void Add(int[] result, int[] array1, int[] array2)
        {
            if (array1.Length != result.Length || array2.Length != result.Length)
                throw new ArgumentOutOfRangeException("Bad arrays given; they need to be the same length.");

            SafeNativeMethods.AddIntArrays(DeviceId, result, array1, array2, array1.Length);
        }

        public int[] Add(int[] array1, int[] array2)
        {
            if (array1.Length != array2.Length)
                throw new ArgumentOutOfRangeException("Bad arrays given; they need to be the same length.");

            var result = new int[array1.Length];
            SafeNativeMethods.AddIntArrays(DeviceId, result, array1, array2, array1.Length);
            return result;
        }

        public void Add(float[] result, float[] array1, float[] array2)
        {
            if (array1.Length != result.Length || array2.Length != result.Length)
                throw new ArgumentOutOfRangeException("Bad arrays given; they need to be the same length.");

            SafeNativeMethods.AddFloatArrays(DeviceId, result, array1, array2, array1.Length);
        }

        public float[] Add(float[] array1, float[] array2)
        {
            if (array1.Length != array2.Length)
                throw new ArgumentOutOfRangeException("Bad arrays given; they need to be the same length.");

            var result = new float[array1.Length];
            SafeNativeMethods.AddFloatArrays(DeviceId, result, array1, array2, array1.Length);
            return result;
        }

        public void Add(long[] result, long[] array1, long[] array2)
        {
            if (array1.Length != result.Length || array2.Length != result.Length)
                throw new ArgumentOutOfRangeException("Bad arrays given; they need to be the same length.");

            SafeNativeMethods.AddLongArrays(DeviceId, result, array1, array2, array1.Length);
        }

        public long[] Add(long[] array1, long[] array2)
        {
            if (array1.Length != array2.Length)
                throw new Exception("Bad arrays given; they need to be the same length.");

            var result = new long[array1.Length];
            SafeNativeMethods.AddLongArrays(DeviceId, result, array1, array2, array1.Length);
            return result;
        }

        public void Add(double[] result, double[] array1, double[] array2)
        {
            if (array1.Length != result.Length || array2.Length != result.Length)
                throw new Exception("Bad arrays given; they need to be the same length.");

            SafeNativeMethods.AddDoubleArrays(DeviceId, result, array1, array2, array1.Length);
        }

        public double[] Add(double[] array1, double[] array2)
        {
            if (array1.Length != array2.Length)
                throw new Exception("Bad arrays given; they need to be the same length.");

            var result = new double[array1.Length];
            SafeNativeMethods.AddDoubleArrays(DeviceId, result, array1, array2, array1.Length);
            return result;
        }

        public void Subtract(int[] result, int[] array1, int[] array2)
        {
            if (array1.Length != result.Length || array2.Length != result.Length)
                throw new ArgumentOutOfRangeException("Bad arrays given; they need to be the same length.");

            SafeNativeMethods.SubtractIntArrays(DeviceId, result, array1, array2, array1.Length);
        }

        public int[] Subtract(int[] array1, int[] array2)
        {
            if (array1.Length != array2.Length)
                throw new ArgumentOutOfRangeException("Bad arrays given; they need to be the same length.");

            var result = new int[array1.Length];
            SafeNativeMethods.SubtractIntArrays(DeviceId, result, array1, array2, array1.Length);
            return result;
        }

        public void Subtract(float[] result, float[] array1, float[] array2)
        {
            if (array1.Length != result.Length || array2.Length != result.Length)
                throw new ArgumentOutOfRangeException("Bad arrays given; they need to be the same length.");

            SafeNativeMethods.SubtractFloatArrays(DeviceId, result, array1, array2, array1.Length);
        }

        public float[] Subtract(float[] array1, float[] array2)
        {
            if (array1.Length != array2.Length)
                throw new ArgumentOutOfRangeException("Bad arrays given; they need to be the same length.");

            var result = new float[array1.Length];
            SafeNativeMethods.SubtractFloatArrays(DeviceId, result, array1, array2, array1.Length);
            return result;
        }

        public void Subtract(long[] result, long[] array1, long[] array2)
        {
            if (array1.Length != result.Length || array2.Length != result.Length)
                throw new ArgumentOutOfRangeException("Bad arrays given; they need to be the same length.");

            SafeNativeMethods.SubtractLongArrays(DeviceId, result, array1, array2, array1.Length);
        }

        public long[] Subtract(long[] array1, long[] array2)
        {
            if (array1.Length != array2.Length)
                throw new Exception("Bad arrays given; they need to be the same length.");

            var result = new long[array1.Length];
            SafeNativeMethods.SubtractLongArrays(DeviceId, result, array1, array2, array1.Length);
            return result;
        }

        public void Subtract(double[] result, double[] array1, double[] array2)
        {
            if (array1.Length != result.Length || array2.Length != result.Length)
                throw new Exception("Bad arrays given; they need to be the same length.");

            SafeNativeMethods.SubtractDoubleArrays(DeviceId, result, array1, array2, array1.Length);
        }

        public double[] Subtract(double[] array1, double[] array2)
        {
            if (array1.Length != array2.Length)
                throw new Exception("Bad arrays given; they need to be the same length.");

            var result = new double[array1.Length];
            SafeNativeMethods.SubtractDoubleArrays(DeviceId, result, array1, array2, array1.Length);
            return result;
        }

        private T[] Flatten<T>(int rows, int columns, T[][] nested_array)
        {
            var flat_array = new T[rows * columns];
            for (int y = 0; y < rows; y++)
            {
                for (int x = 0; x < columns; x++)
                {
                    flat_array[(y * rows) + x] = nested_array[y][x];
                }
            }

            return flat_array;
        }

        private T[][] Unflatten<T>(int rows, int columns, T[] flat_array)
        {
            var nested_array = new T[rows][];
            for (int y = 0; y < rows; y++)
            {
                nested_array[y] = new T[columns];

                for (int x = 0; x < columns; x++)
                {
                    nested_array[y][x] = flat_array[(y * rows) + x];
                }
            }

            return nested_array;
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

        public float[][] Multiply(
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

            // .NET does not support marshaling nested arrays between C++ and e.g. C#.
            // If you try, you will get the error message, "There is no marshaling support for nested arrays."
            // Further, the cuBLAS function cublasSgemm/cublasDgemm does not have pointer-to-pointers as arguments (e.g., float**), so we cannot
            // supply a multidimensional array anyway.
            // The solution: flatten arrays so that they can passed to CudaSharperLibrary, and then unflatten whatever it passes back.
            var d_a = Flatten(a.Length, a[0].Length, a);
            var d_b = Flatten(b.Length, b[0].Length, b);

            // Despite the definition above, this will return the correct size for C. Go figure.
            var d_c = new float[a.Length * b.Length];

            var transa_op = (uint)a_op;
            var transb_op = (uint)b_op;

            SafeNativeMethods.MatrixMultiplyFloat(
                (uint)DeviceId,
                transa_op, transb_op,
                a.Length, b[0].Length, a[0].Length,
                alpha,
                d_a,
                d_b,
                beta,
                d_c);

            return Unflatten(a.Length, b.Length, d_c);
        }

        public double[][] Multiply(
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

            // .NET does not support marshaling nested arrays between C++ and e.g. C#.
            // If you try, you will get the error message, "There is no marshaling support for nested arrays."
            // Further, the cuBLAS function cublasSgemm does not take pointer-to-pointers (e.g., float**), so we cannot
            // supply a multidimensional array anyway.
            // The solution: flatten arrays so that they can passed to CudaSharperLibrary, and then unflatten whatever it passes back.
            var d_a = Flatten(a.Length, a[0].Length, a);
            var d_b = Flatten(b.Length, b[0].Length, b);

            // Despite the definition above, this will return the correct size for C. Go figure.
            var d_c = new double[a.Length * b.Length];

            var transa_op = (uint)a_op;
            var transb_op = (uint)b_op;

            SafeNativeMethods.MatrixMultiplyDouble(
                (uint)DeviceId,
                transa_op, transb_op,
                a.Length, b[0].Length, a[0].Length,
                alpha,
                d_a,
                d_b,
                beta,
                d_c);

            return Unflatten(a.Length, b.Length, d_c);
        }
    }
}
