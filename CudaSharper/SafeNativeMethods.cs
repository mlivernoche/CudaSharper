using System;
using System.Runtime.InteropServices;
using System.Text;

/*
 * SaveNativeMethods - these call directly to methods specified in CudaSharperLibrary.dll.
 * The types used here must be CLS-compliant and they must be specific.
 * They must align with what is defined in the C++ code; e.g., int64_t must be mirrored with Int64.
 * These methods are as raw as possible - any sort of type translation should be done in DTM.
 * 
 * Example: CUDA provides cudaError_t, which is an error code that each function returns. It is an int enum.
 * Enums are not blittable, but ints are. The C++ code takes the cudaError_t, casts it to an int, and returns it.
 * This is why all of these methods have an int return value. DTM is where that int is then cast back to an enum,
 * which is defined in C# as CudaError.
 * 
 * Rules for adding a method:
 * - DO use Int32 instead of int and Int64 instead of long.
 * - DO make ALL methods return Int32 (and no tuples).
 * - DO make ALL methods internal (absolutely no public methods).
 * - DO make device_id the first parameter.
 * - DO make device_id Int32.
 * - DO make the size of the array follow the array (e.g., Method(float[] arr, Int32 arr_size)).
 * - DO NOT use uint, ulong, etc.
 * - DO NOT generics.
 * - DO NOT use Int16 or byte/sbyte.
 */

namespace CudaSharper
{
    internal static class SafeNativeMethods
    {
        [DllImport("CudaSharperLibrary.dll")]
        internal static extern IntPtr CreateRandomClass(Int32 device_id, Int64 amount_of_numbers);

        [DllImport("CudaSharperLibrary.dll")]
        internal static extern void DisposeRandomClass(IntPtr cuda_rand);

        [DllImport("CudaSharperLibrary.dll")]
        internal static extern IntPtr CreateArrayClass(Int32 device_id, Int64 amount_of_numbers);

        [DllImport("CudaSharperLibrary.dll")]
        internal static extern void DisposeArrayClass(IntPtr cuda_rand);

        [DllImport("CudaSharperLibrary.dll")]
        internal static extern IntPtr CreateStatClass(Int32 device_id, Int64 amount_of_numbers);

        [DllImport("CudaSharperLibrary.dll")]
        internal static extern void DisposeStatClass(IntPtr cuda_rand);

        #region CudaDevice.cs
        [DllImport(
            "CudaSharperLibrary.dll",
            CallingConvention = CallingConvention.Cdecl,
            CharSet = CharSet.Ansi,
            BestFitMapping = false,
            ThrowOnUnmappableChar = true)]
        internal static extern Int32 GetCudaDeviceName(Int32 device_id, StringBuilder device_name_ptr);
        #endregion

        #region CudaSettings.cs
        [DllImport("CudaSharperLibrary.dll", EntryPoint = "GetCudaDeviceCount")]
        internal static extern Int32 GetCudaDeviceCount();

        [DllImport("CudaSharperLibrary.dll", EntryPoint = "ResetCudaDevice")]
        internal static extern Int32 ResetCudaDevice();
        #endregion

        #region CuRand.cs
        [DllImport("CudaSharperLibrary.dll")]
        internal static extern Int32 UniformRand(IntPtr cuda_rand, float[] result, Int64 amount_of_numbers);

        [DllImport("CudaSharperLibrary.dll")]
        internal static extern Int32 UniformRandDouble(IntPtr cuda_rand, double[] result, Int64 amount_of_numbers);

        [DllImport("CudaSharperLibrary.dll")]
        internal static extern Int32 LogNormalRand(IntPtr cuda_rand, float[] result, Int64 amount_of_numbers, float mean, float stddev);

        [DllImport("CudaSharperLibrary.dll")]
        internal static extern Int32 LogNormalRandDouble(IntPtr cuda_rand, double[] result, Int64 amount_of_numbers, float mean, float stddev);

        [DllImport("CudaSharperLibrary.dll")]
        internal static extern Int32 NormalRand(IntPtr cuda_rand, float[] result, Int64 amount_of_numbers);

        [DllImport("CudaSharperLibrary.dll")]
        internal static extern Int32 NormalRandDouble(IntPtr cuda_rand, double[] result, Int64 amount_of_numbers);

        [DllImport("CudaSharperLibrary.dll")]
        internal static extern Int32 PoissonRand(IntPtr cuda_rand, Int32[] result, Int64 amount_of_numbers, double lambda);
        #endregion

        #region CuArray.cs
        [DllImport("CudaSharperLibrary.dll")]
        internal static extern Int32 AddIntArrays(IntPtr cuarray, Int32[] result, Int32[] array1, Int32[] array2, Int64 length);

        [DllImport("CudaSharperLibrary.dll")]
        internal static extern Int32 AddLongArrays(IntPtr cuarray, Int64[] result, Int64[] array1, Int64[] array2, Int64 length);

        [DllImport("CudaSharperLibrary.dll")]
        internal static extern Int32 AddFloatArrays(IntPtr cuarray, float[] result, float[] array1, float[] array2, Int64 length);

        [DllImport("CudaSharperLibrary.dll")]
        internal static extern Int32 AddDoubleArrays(IntPtr cuarray, double[] result, double[] array1, double[] array2, Int64 length);

        [DllImport("CudaSharperLibrary.dll")]
        internal static extern Int32 SubtractIntArrays(IntPtr cuarray, Int32[] result, Int32[] array1, Int32[] array2, Int64 length);

        [DllImport("CudaSharperLibrary.dll")]
        internal static extern Int32 SubtractFloatArrays(IntPtr cuarray, float[] result, float[] array1, float[] array2, Int64 length);

        [DllImport("CudaSharperLibrary.dll")]
        internal static extern Int32 SubtractLongArrays(IntPtr cuarray, Int64[] result, Int64[] array1, Int64[] array2, Int64 length);

        [DllImport("CudaSharperLibrary.dll")]
        internal static extern Int32 SubtractDoubleArrays(IntPtr cuarray, double[] result, double[] array1, double[] array2, Int64 length);

        [DllImport("CudaSharperLibrary.dll")]
        internal static extern Int32 MatrixMultiplyFloat(
            Int32 device_id,
            Int32 transa_op, Int32 transb_op,
            Int32 m, Int32 n, Int32 k,
            float alpha,
            float[] a,
            float[] b,
            float beta,
            float[] c);

        [DllImport("CudaSharperLibrary.dll")]
        internal static extern Int32 MatrixMultiplyDouble(
            Int32 device_id,
            Int32 transa_op, Int32 transb_op,
            Int32 m, Int32 n, Int32 k,
            double alpha,
            double[] a,
            double[] b,
            double beta,
            double[] c);
        #endregion

        #region cuStats.cs
        [DllImport("CudaSharperLibrary.dll")]
        internal static extern Int32 SampleStandardDeviationFloat(IntPtr custat, ref double result, float[] sample, Int64 sample_size, float mean);

        [DllImport("CudaSharperLibrary.dll")]
        internal static extern Int32 SampleStandardDeviationDouble(IntPtr custat, ref double result, double[] sample, Int64 sample_size, double mean);

        [DllImport("CudaSharperLibrary.dll")]
        internal static extern Int32 StandardDeviationFloat(IntPtr custat, ref double result, float[] sample, Int64 sample_size, float mean);

        [DllImport("CudaSharperLibrary.dll")]
        internal static extern Int32 StandardDeviationDouble(IntPtr custat, ref double result, double[] sample, Int64 sample_size, double mean);

        [DllImport("CudaSharperLibrary.dll")]
        internal static extern Int32 SampleCovarianceFloat(IntPtr custat, ref double result, float[] x_array, float x_mean, float[] y_array, float y_mean, Int64 sample_size);

        [DllImport("CudaSharperLibrary.dll")]
        internal static extern Int32 SampleCovarianceDouble(IntPtr custat, ref double result, double[] x_array, double x_mean, double[] y_array, double y_mean, Int64 sample_size);

        [DllImport("CudaSharperLibrary.dll")]
        internal static extern Int32 CovarianceFloat(IntPtr custat, ref double result, float[] x_array, float x_mean, float[] y_array, float y_mean, Int64 sample_size);

        [DllImport("CudaSharperLibrary.dll")]
        internal static extern Int32 CovarianceDouble(IntPtr custat, ref double result, double[] x_array, double x_mean, double[] y_array, double y_mean, Int64 sample_size);

        [DllImport("CudaSharperLibrary.dll")]
        internal static extern Int32 PearsonCorrelationFloat(IntPtr custat, ref double result, float[] x_array, float x_mean, float[] y_array, float y_mean, Int64 sample_size);

        [DllImport("CudaSharperLibrary.dll")]
        internal static extern Int32 PearsonCorrelationDouble(IntPtr custat, ref double result, double[] x_array, double x_mean, double[] y_array, double y_mean, Int64 sample_size);
        #endregion
    }
}
