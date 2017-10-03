using System.Runtime.InteropServices;
using System.Text;

/*
 * CudaSharper - a wrapper for CUDA-accelerated functions. CudaSharper is not intended to write CUDA in C#, but rather a
 * library that allows one to easily use CUDA-accelerated functions without having to directly interact with the device.
 * This file acts as a wrapper for CudaSharperLibrary.dll, which is required for these functions to run. CudaSharperLibrary.dll
 * is the actual CUDA C code compiled as a C++/CLI assembly; however, it is unmanaged and therefore requires this wrapper to be used
 * in C# projects.
 * 
 * Current Classes:
 * - CudaSettings: Initialization class.
 * - Cuda: Array functions, such as SplitArray and AddArrays
 * - CuRand: cuRAND functions; allows the generation of pseudorandom number sets using uniform, normal, or poisson distributions.
 */

namespace CudaSharper
{
    internal static partial class SafeNativeMethods
    {
        #region CudaDevice.cs
        [DllImport("CudaSharperLibrary.dll", CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Unicode, ThrowOnUnmappableChar = true)]
        internal static extern void GetCudaDeviceName(int device_id, StringBuilder device_name_ptr);
        #endregion

        #region CudaSettings.cs
        [DllImport("CudaSharperLibrary.dll")]
        internal static extern int GetCudaDeviceCount();
        #endregion

        #region CuRand.cs
        [DllImport("CudaSharperLibrary.dll")]
        internal static extern void UniformRand(int device_id, int amount_of_numbers, float[] result);

        [DllImport("CudaSharperLibrary.dll")]
        internal static extern void UniformRandDouble(int device_id, int amount_of_numbers, double[] result);

        [DllImport("CudaSharperLibrary.dll")]
        internal static extern void LogNormalRand(int device_id, int amount_of_numbers, float[] result, float mean, float stddev);

        [DllImport("CudaSharperLibrary.dll")]
        internal static extern void LogNormalRandDouble(int device_id, int amount_of_numbers, double[] result, float mean, float stddev);

        [DllImport("CudaSharperLibrary.dll")]
        internal static extern void NormalRand(int device_id, int amount_of_numbers, float[] result);

        [DllImport("CudaSharperLibrary.dll")]
        internal static extern void NormalRandDouble(int device_id, int amount_of_numbers, double[] result);

        [DllImport("CudaSharperLibrary.dll")]
        internal static extern void PoissonRand(int device_id, int amount_of_numbers, int[] result, double lambda);
        #endregion

        #region CuArray.cs
        [DllImport("CudaSharperLibrary.dll")]
        internal static extern void SplitIntArray(int device_id, int[] src, int[] array1, int[] array2, uint length, uint split_index);

        [DllImport("CudaSharperLibrary.dll")]
        internal static extern void SplitLongArray(int device_id, long[] src, long[] array1, long[] array2, uint length, uint split_index);

        [DllImport("CudaSharperLibrary.dll")]
        internal static extern void SplitFloatArray(int device_id, float[] src, float[] array1, float[] array2, uint length, uint split_index);

        [DllImport("CudaSharperLibrary.dll")]
        internal static extern void SplitDoubleArray(int device_id, double[] src, double[] array1, double[] array2, uint length, uint split_index);

        [DllImport("CudaSharperLibrary.dll")]
        internal static extern void MergeIntArrays(int device_id, int[] result, int[] array1, int[] array2, uint array1_length, uint array2_length);

        [DllImport("CudaSharperLibrary.dll")]
        internal static extern void MergeLongArrays(int device_id, long[] result, long[] array1, long[] array2, uint array1_length, uint array2_length);

        [DllImport("CudaSharperLibrary.dll")]
        internal static extern void MergeFloatArrays(int device_id, float[] result, float[] array1, float[] array2, int array1_length, int array2_length);

        [DllImport("CudaSharperLibrary.dll")]
        internal static extern void MergeDoubleArrays(int device_id, double[] result, double[] array1, double[] array2, int array1_length, int array2_length);

        [DllImport("CudaSharperLibrary.dll")]
        internal static extern void AddIntArrays(int device_id, int[] result, int[] array1, int[] array2, int length);

        [DllImport("CudaSharperLibrary.dll")]
        internal static extern void AddLongArrays(int device_id, long[] result, long[] array1, long[] array2, int length);

        [DllImport("CudaSharperLibrary.dll")]
        internal static extern void AddFloatArrays(int device_id, float[] result, float[] array1, float[] array2, int length);

        [DllImport("CudaSharperLibrary.dll")]
        internal static extern void AddDoubleArrays(int device_id, double[] result, double[] array1, double[] array2, int length);

        [DllImport("CudaSharperLibrary.dll")]
        internal static extern void SubtractIntArrays(int device_id, int[] result, int[] array1, int[] array2, int length);

        [DllImport("CudaSharperLibrary.dll")]
        internal static extern void SubtractFloatArrays(int device_id, float[] result, float[] array1, float[] array2, int length);

        [DllImport("CudaSharperLibrary.dll")]
        internal static extern void SubtractLongArrays(int device_id, long[] result, long[] array1, long[] array2, int length);

        [DllImport("CudaSharperLibrary.dll")]
        internal static extern void SubtractDoubleArrays(int device_id, double[] result, double[] array1, double[] array2, int length);

        [DllImport("CudaSharperLibrary.dll")]
        internal static extern void MatrixMultiplyFloat(
            uint device_id,
            uint transa_op, uint transb_op,
            int m, int n, int k,
            float alpha,
            float[] a,
            float[] b,
            float beta,
            float[] c);

        [DllImport("CudaSharperLibrary.dll")]
        internal static extern void MatrixMultiplyDouble(
            uint device_id,
            uint transa_op, uint transb_op,
            int m, int n, int k,
            double alpha,
            double[] a,
            double[] b,
            double beta,
            double[] c);
        #endregion

        #region cuStats.cs
        [DllImport("CudaSharperLibrary.dll")]
        internal static extern double SampleStandardDeviationFloat(uint device_id, float[] sample, ulong sample_size, double mean);

        [DllImport("CudaSharperLibrary.dll")]
        internal static extern double SampleStandardDeviationDouble(uint device_id, double[] sample, ulong sample_size, double mean);

        [DllImport("CudaSharperLibrary.dll")]
        internal static extern double StandardDeviationFloat(uint device_id, float[] sample, ulong sample_size, double mean);

        [DllImport("CudaSharperLibrary.dll")]
        internal static extern double StandardDeviationDouble(uint device_id, double[] sample, ulong sample_size, double mean);

        [DllImport("CudaSharperLibrary.dll")]
        internal static extern double SampleCovarianceFloat(uint device_id, float[] x_array, double x_mean, float[] y_array, double y_mean, ulong sample_size);

        [DllImport("CudaSharperLibrary.dll")]
        internal static extern double SampleCovarianceDouble(uint device_id, double[] x_array, double x_mean, double[] y_array, double y_mean, ulong sample_size);

        [DllImport("CudaSharperLibrary.dll")]
        internal static extern double CovarianceDouble(uint device_id, double[] x_array, double x_mean, double[] y_array, double y_mean, ulong sample_size);

        [DllImport("CudaSharperLibrary.dll")]
        internal static extern double CovarianceFloat(uint device_id, float[] x_array, double x_mean, float[] y_array, double y_mean, ulong sample_size);

        [DllImport("CudaSharperLibrary.dll")]
        internal static extern double PearsonCorrelationFloat(uint device_id, float[] x_array, double x_mean, float[] y_array, double y_mean, ulong sample_size);

        [DllImport("CudaSharperLibrary.dll")]
        internal static extern double PearsonCorrelationDouble(uint device_id, double[] x_array, double x_mean, double[] y_array, double y_mean, ulong sample_size);
        #endregion
    }
}
