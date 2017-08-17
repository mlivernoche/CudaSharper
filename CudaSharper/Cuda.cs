using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

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
    public static class CudaSettings
    {
        private static bool workingDirSet { get; set; } = false;
        public static int DeviceId { get; set; }

        public static void Load(string working_directory)
        {
            if (workingDirSet)
                return;

            Environment.SetEnvironmentVariable("PATH", Environment.GetEnvironmentVariable("PATH") + AppDomain.CurrentDomain.BaseDirectory);
            workingDirSet = true;
        }

        public static void Load()
        {
            Load(AppDomain.CurrentDomain.BaseDirectory);
        }
    }

    public static class Cuda
    {
        [DllImport("CudaSharperLibrary.dll")]
        private static extern void SplitIntArray(int device_id, int[] src, int[] array1, int[] array2, int length, int split_index);

        public static void SplitArray(int[] src, int[] array1, int[] array2, int length, int split_index)
        {
            SplitIntArray(CudaSettings.DeviceId, src, array1, array2, length, split_index);
        }

        public static (IEnumerable<int>, IEnumerable<int>) SplitArray(int[] src, int length, int split_index)
        {
            var array1 = new int[length];
            var array2 = new int[length];
            SplitIntArray(CudaSettings.DeviceId, src, array1, array2, length, split_index);
            return (array1, array2);
        }

        [DllImport("CudaSharperLibrary.dll")]
        private static extern void AddIntArrays(int device_id, int[] result, int[] array1, int[] array2, int length);

        public static void AddArrays(int[] result, int[] array1, int[] array2, int length)
        {
            AddIntArrays(CudaSettings.DeviceId, result, array1, array2, length);
        }

        public static IEnumerable<int> AddArrays(int[] array1, int[] array2, int length)
        {
            var result = new int[length];
            AddIntArrays(CudaSettings.DeviceId, result, array1, array2, length);
            return result;
        }

        [DllImport("CudaSharperLibrary.dll")]
        private static extern void AddFloatArrays(int device_id, float[] result, float[] array1, float[] array2, int length);

        public static void AddArrays(float[] result, float[] array1, float[] array2, int length)
        {
            AddFloatArrays(CudaSettings.DeviceId, result, array1, array2, length);
        }

        public static IEnumerable<float> AddArrays(float[] array1, float[] array2, int length)
        {
            var result = new float[length];
            AddFloatArrays(CudaSettings.DeviceId, result, array1, array2, length);
            return result;
        }

        [DllImport("CudaSharperLibrary.dll")]
        private static extern void AddLongArrays(int device_id, long[] result, long[] array1, long[] array2, int length);

        public static void AddArrays(long[] result, long[] array1, long[] array2, int length)
        {
            AddLongArrays(CudaSettings.DeviceId, result, array1, array2, length);
        }

        public static IEnumerable<long> AddArrays(long[] array1, long[] array2, int length)
        {
            var result = new long[length];
            AddLongArrays(CudaSettings.DeviceId, result, array1, array2, length);
            return result;
        }

        [DllImport("CudaSharperLibrary.dll")]
        private static extern void AddDoubleArrays(int device_id, double[] result, double[] array1, double[] array2, int length);

        public static void AddArrays(double[] result, double[] array1, double[] array2, int length)
        {
            AddDoubleArrays(CudaSettings.DeviceId, result, array1, array2, length);
        }

        public static IEnumerable<double> AddArrays(double[] array1, double[] array2, int length)
        {
            var result = new double[length];
            AddDoubleArrays(CudaSettings.DeviceId, result, array1, array2, length);
            return result;
        }
    }

    public static class CuRand
    {
        [DllImport("CudaSharperLibrary.dll")]
        private static extern void UniformRand(int device_id, int amount_of_numbers, float[] result);

        public static void GenerateUniformDistribution(int amount_of_numbers, float[] result)
        {
            UniformRand(CudaSettings.DeviceId, amount_of_numbers, result);
        }

        public static IEnumerable<float> GenerateUniformDistribution(int amount_of_numbers)
        {
            var result = new float[amount_of_numbers];
            UniformRand(CudaSettings.DeviceId, amount_of_numbers, result);
            return result;
        }

        [DllImport("CudaSharperLibrary.dll")]
        private static extern void NormalRand(int device_id, int amount_of_numbers, float[] result);

        public static void GenerateNormalDistribution(int amount_of_numbers, float[] result)
        {
            NormalRand(CudaSettings.DeviceId, amount_of_numbers, result);
        }

        public static IEnumerable<float> GenerateNormalDistribution(int amount_of_numbers)
        {
            var result = new float[amount_of_numbers];
            NormalRand(CudaSettings.DeviceId, amount_of_numbers, result);
            return result;
        }

        [DllImport("CudaSharperLibrary.dll")]
        private static extern void PoissonRand(int device_id, int amount_of_numbers, int[] result, double lambda);

        public static void GeneratePoissonDistribution(int amount_of_numbers, int[] result, double lambda)
        {
            PoissonRand(CudaSettings.DeviceId, amount_of_numbers, result, lambda);
        }

        public static IEnumerable<int> GeneratePoissonDistribution(int amount_of_numbers, double lambda)
        {
            var result = new int[amount_of_numbers];
            PoissonRand(CudaSettings.DeviceId, amount_of_numbers, result, lambda);
            return result;
        }
    }
}
