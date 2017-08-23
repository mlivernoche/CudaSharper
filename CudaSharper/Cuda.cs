using System;
using System.Collections.Generic;
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
    public static class CudaSettings
    {
        private static bool workingDirSet { get; set; } = false;

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

    internal interface ICudaInfo
    {
        int CudaDevicesCount();
        string GetCudaDeviceName(int device_id);
        int DeviceId { get; }
    }

    public class CudaInfo
    {
        [DllImport("CudaSharperLibrary.dll")]
        private static extern int GetCudaDeviceCount();

        public int CudaDevicesCount()
        {
            return GetCudaDeviceCount();
        }

        [DllImport("CudaSharperLibrary.dll", CallingConvention = CallingConvention.Cdecl)]
        private static extern void GetCudaDeviceName(int device_id, StringBuilder device_name_ptr);

        public string GetCudaDeviceName(int device_id)
        {
            StringBuilder device_name = new StringBuilder(256);
            GetCudaDeviceName(device_id, device_name);
            return device_name.ToString();
        }
    }

    internal interface ICuda : ICudaInfo
    {
        /*
         * SplitArray:
         * - int[], (IEnumerable<int>, IEnumerable<int>)
         * - long[], (IEnumerable<float>, IEnumerable<float>)
         * - float[], (IEnumerable<float>, IEnumerable<float>)
         * - double[], (IEnumerable<double>, IEnumerable<double>)
         */
        void SplitArray(int[] src, int[] array1, int[] array2, int length, int split_index);
        (IEnumerable<int>, IEnumerable<int>) SplitArray(int[] src, int length, int split_index);

        void SplitArray(long[] src, long[] array1, long[] array2, int length, int split_index);
        (IEnumerable<long>, IEnumerable<long>) SplitArray(long[] src, int length, int split_index);

        void SplitArray(float[] src, float[] array1, float[] array2, int length, int split_index);
        (IEnumerable<float>, IEnumerable<float>) SplitArray(float[] src, int length, int split_index);

        void SplitArray(double[] src, double[] array1, double[] array2, int length, int split_index);
        (IEnumerable<double>, IEnumerable<double>) SplitArray(double[] src, int length, int split_index);

        /*
         * MergeArrays:
         * - int[], IEnumerable<int>
         * - long[], IEnumerable<float>
         * - float[], IEnumerable<float>
         * - double[], IEnumerable<double>
         */
        void MergeArrays(int[] result, int[] array1, int[] array2);
        IEnumerable<int> MergeArrays(int[] array1, int[] array2);

        void MergeArrays(long[] result, long[] array1, long[] array2);
        IEnumerable<long> MergeArrays(long[] array1, long[] array2);

        void MergeArrays(float[] result, float[] array1, float[] array2);
        IEnumerable<float> MergeArrays(float[] array1, float[] array2);

        void MergeArrays(double[] result, double[] array1, double[] array2);
        IEnumerable<double> MergeArrays(double[] array1, double[] array2);

        /*
         * MergeArrays:
         * - int[], IEnumerable<int>
         * - long[], IEnumerable<float>
         * - float[], IEnumerable<float>
         * - double[], IEnumerable<double>
         */
        void AddArrays(int[] result, int[] array1, int[] array2);
        IEnumerable<int> AddArrays(int[] array1, int[] array2);

        void AddArrays(long[] result, long[] array1, long[] array2);
        IEnumerable<long> AddArrays(long[] array1, long[] array2);

        void AddArrays(float[] result, float[] array1, float[] array2);
        IEnumerable<float> AddArrays(float[] array1, float[] array2);

        void AddArrays(double[] result, double[] array1, double[] array2);
        IEnumerable<double> AddArrays(double[] array1, double[] array2);
    }

    public class Cuda : CudaInfo, ICuda
    {
        public int DeviceId { get; }

        static Cuda()
        {
            CudaSettings.Load();
        }

        public Cuda(int device_id)
        {
            if((device_id + 1) > CudaDevicesCount())
                throw new Exception("Bad DeviceId provided: not enough CUDA-enabled devices available.");

            DeviceId = device_id;
        }

        [DllImport("CudaSharperLibrary.dll")]
        private static extern void SplitIntArray(int device_id, int[] src, int[] array1, int[] array2, int length, int split_index);

        public void SplitArray(int[] src, int[] array1, int[] array2, int length, int split_index)
        {
            SplitIntArray(DeviceId, src, array1, array2, length, split_index);
        }

        public (IEnumerable<int>, IEnumerable<int>) SplitArray(int[] src, int length, int split_index)
        {
            var array1 = new int[split_index];
            var array2 = new int[length - split_index];
            SplitIntArray(DeviceId, src, array1, array2, length, split_index);
            return (array1, array2);
        }

        [DllImport("CudaSharperLibrary.dll")]
        private static extern void SplitLongArray(int device_id, long[] src, long[] array1, long[] array2, int length, int split_index);

        public void SplitArray(long[] src, long[] array1, long[] array2, int length, int split_index)
        {
            SplitLongArray(DeviceId, src, array1, array2, length, split_index);
        }

        public (IEnumerable<long>, IEnumerable<long>) SplitArray(long[] src, int length, int split_index)
        {
            var array1 = new long[split_index];
            var array2 = new long[length - split_index];
            SplitLongArray(DeviceId, src, array1, array2, length, split_index);
            return (array1, array2);
        }

        [DllImport("CudaSharperLibrary.dll")]
        private static extern void SplitFloatArray(int device_id, float[] src, float[] array1, float[] array2, int length, int split_index);

        public void SplitArray(float[] src, float[] array1, float[] array2, int length, int split_index)
        {
            SplitFloatArray(DeviceId, src, array1, array2, length, split_index);
        }

        public (IEnumerable<float>, IEnumerable<float>) SplitArray(float[] src, int length, int split_index)
        {
            var array1 = new float[split_index];
            var array2 = new float[length - split_index];
            SplitFloatArray(DeviceId, src, array1, array2, length, split_index);
            return (array1, array2);
        }

        [DllImport("CudaSharperLibrary.dll")]
        private static extern void SplitDoubleArray(int device_id, double[] src, double[] array1, double[] array2, int length, int split_index);

        public void SplitArray(double[] src, double[] array1, double[] array2, int length, int split_index)
        {
            SplitDoubleArray(DeviceId, src, array1, array2, length, split_index);
        }

        public (IEnumerable<double>, IEnumerable<double>) SplitArray(double[] src, int length, int split_index)
        {
            var array1 = new double[split_index];
            var array2 = new double[length - split_index];
            SplitDoubleArray(DeviceId, src, array1, array2, length, split_index);
            return (array1, array2);
        }

        [DllImport("CudaSharperLibrary.dll")]
        private static extern void MergeIntArrays(int device_id, int[] result, int[] array1, int[] array2, int array1_length, int array2_length);

        public void MergeArrays(int[] result, int[] array1, int[] array2)
        {
            MergeIntArrays(DeviceId, result, array1, array2, array1.Length, array2.Length);
        }

        public IEnumerable<int> MergeArrays(int[] array1, int[] array2)
        {
            var result = new int[array1.Length + array2.Length];
            MergeIntArrays(DeviceId, result, array1, array2, array1.Length, array2.Length);
            return result;
        }

        [DllImport("CudaSharperLibrary.dll")]
        private static extern void MergeLongArrays(int device_id, long[] result, long[] array1, long[] array2, int array1_length, int array2_length);

        public void MergeArrays(long[] result, long[] array1, long[] array2)
        {
            MergeLongArrays(DeviceId, result, array1, array2, array1.Length, array2.Length);
        }

        public IEnumerable<long> MergeArrays(long[] array1, long[] array2)
        {
            var result = new long[array1.Length + array2.Length];
            MergeLongArrays(DeviceId, result, array1, array2, array1.Length, array2.Length);
            return result;
        }

        [DllImport("CudaSharperLibrary.dll")]
        private static extern void MergeFloatArrays(int device_id, float[] result, float[] array1, float[] array2, int array1_length, int array2_length);

        public void MergeArrays(float[] result, float[] array1, float[] array2)
        {
            MergeFloatArrays(DeviceId, result, array1, array2, array1.Length, array2.Length);
        }

        public IEnumerable<float> MergeArrays(float[] array1, float[] array2)
        {
            var result = new float[array1.Length + array2.Length];
            MergeFloatArrays(DeviceId, result, array1, array2, array1.Length, array2.Length);
            return result;
        }

        [DllImport("CudaSharperLibrary.dll")]
        private static extern void MergeDoubleArrays(int device_id, double[] result, double[] array1, double[] array2, int array1_length, int array2_length);

        public void MergeArrays(double[] result, double[] array1, double[] array2)
        {
            MergeDoubleArrays(DeviceId, result, array1, array2, array1.Length, array2.Length);
        }

        public IEnumerable<double> MergeArrays(double[] array1, double[] array2)
        {
            var result = new double[array1.Length + array2.Length];
            MergeDoubleArrays(DeviceId, result, array1, array2, array1.Length, array2.Length);
            return result;
        }

        [DllImport("CudaSharperLibrary.dll")]
        private static extern void AddIntArrays(int device_id, int[] result, int[] array1, int[] array2, int length);

        public void AddArrays(int[] result, int[] array1, int[] array2)
        {
            if (array1.Length != result.Length || array2.Length != result.Length) throw new Exception("Bad arrays given; they need to be the same length.");
            AddIntArrays(DeviceId, result, array1, array2, array1.Length);
        }

        public IEnumerable<int> AddArrays(int[] array1, int[] array2)
        {
            if (array1.Length != array2.Length) throw new Exception("Bad arrays given; they need to be the same length.");
            var result = new int[array1.Length];
            AddIntArrays(DeviceId, result, array1, array2, array1.Length);
            return result;
        }

        [DllImport("CudaSharperLibrary.dll")]
        private static extern void AddFloatArrays(int device_id, float[] result, float[] array1, float[] array2, int length);

        public void AddArrays(float[] result, float[] array1, float[] array2)
        {
            if (array1.Length != result.Length || array2.Length != result.Length) throw new Exception("Bad arrays given; they need to be the same length.");
            AddFloatArrays(DeviceId, result, array1, array2, array1.Length);
        }

        public IEnumerable<float> AddArrays(float[] array1, float[] array2)
        {
            if (array1.Length != array2.Length) throw new Exception("Bad arrays given; they need to be the same length.");
            var result = new float[array1.Length];
            AddFloatArrays(DeviceId, result, array1, array2, array1.Length);
            return result;
        }

        [DllImport("CudaSharperLibrary.dll")]
        private static extern void AddLongArrays(int device_id, long[] result, long[] array1, long[] array2, int length);

        public void AddArrays(long[] result, long[] array1, long[] array2)
        {
            if (array1.Length != result.Length || array2.Length != result.Length) throw new Exception("Bad arrays given; they need to be the same length.");
            AddLongArrays(DeviceId, result, array1, array2, array1.Length);
        }

        public IEnumerable<long> AddArrays(long[] array1, long[] array2)
        {
            if (array1.Length != array2.Length) throw new Exception("Bad arrays given; they need to be the same length.");
            var result = new long[array1.Length];
            AddLongArrays(DeviceId, result, array1, array2, array1.Length);
            return result;
        }

        [DllImport("CudaSharperLibrary.dll")]
        private static extern void AddDoubleArrays(int device_id, double[] result, double[] array1, double[] array2, int length);

        public void AddArrays(double[] result, double[] array1, double[] array2)
        {
            if (array1.Length != result.Length || array2.Length != result.Length) throw new Exception("Bad arrays given; they need to be the same length.");
            AddDoubleArrays(DeviceId, result, array1, array2, array1.Length);
        }

        public IEnumerable<double> AddArrays(double[] array1, double[] array2)
        {
            if (array1.Length != array2.Length) throw new Exception("Bad arrays given; they need to be the same length.");
            var result = new double[array1.Length];
            AddDoubleArrays(DeviceId, result, array1, array2, array1.Length);
            return result;
        }
    }

    internal interface ICuRand : ICudaInfo
    {
        void GenerateUniformDistribution(int amount_of_numbers, float[] result);
        IEnumerable<float> GenerateUniformDistribution(int amount_of_numbers);

        void GenerateUniformDistributionDP(int amount_of_numbers, double[] result);
        IEnumerable<double> GenerateUniformDistributionDP(int amount_of_numbers);

        void GenerateNormalDistribution(int amount_of_numbers, float[] result);
        IEnumerable<float> GenerateNormalDistribution(int amount_of_numbers);

        void GenerateNormalDistributionDP(int amount_of_numbers, double[] result);
        IEnumerable<double> GenerateNormalDistributionDP(int amount_of_numbers);

        void GenerateLogNormalDistribution(int amount_of_numbers, float[] result, float mean, float stddev);
        IEnumerable<float> GenerateLogNormalDistribution(int amount_of_numbers, float mean, float stddev);

        void GenerateLogNormalDistributionDP(int amount_of_numbers, double[] result, float mean, float stddev);
        IEnumerable<double> GenerateLogNormalDistributionDP(int amount_of_numbers, float mean, float stddev);

        void GeneratePoissonDistribution(int amount_of_numbers, int[] result, double lambda);
        IEnumerable<int> GeneratePoissonDistribution(int amount_of_numbers, double lambda);
    }

    public class CuRand : CudaInfo, ICuRand
    {
        public int DeviceId { get; }

        static CuRand()
        {
            CudaSettings.Load();
        }

        public CuRand(int device_id)
        {
            if ((device_id + 1) > CudaDevicesCount())
                throw new Exception("Bad DeviceId provided: not enough CUDA-enabled devices available. Devices available: " + CudaDevicesCount() + ". DeviceId: " + device_id);

            DeviceId = device_id;
        }

        [DllImport("CudaSharperLibrary.dll")]
        private static extern void UniformRand(int device_id, int amount_of_numbers, float[] result);

        /// <summary>
        /// Generates random numbers using XORWOW and a uniform distribution. This method utilizies the single-precision (FP32) capabilities of the GPU. If you need higher precision,
        /// there is a double-precision (FP64) version available; however, performance will be much worse, depending on the GPU being used.
        /// </summary>
        /// <param name="amount_of_numbers">The amount of random numbers to generate.</param>
        /// <param name="result">The array that will hold the random numbers (in memory for the CPU to use).</param>
        public void GenerateUniformDistribution(int amount_of_numbers, float[] result)
        {
            UniformRand(DeviceId, amount_of_numbers, result);
        }

        /// <summary>
        /// Generates random numbers using XORWOW and a uniform distribution. This method utilizies the single-precision (FP32) capabilities of the GPU. If you need higher precision,
        /// there is a double-precision (FP64) version available; however, performance will be much worse, depending on the GPU being used.
        /// </summary>
        /// <param name="amount_of_numbers">The amount of random numbers to generate.</param>
        /// <returns>An IEnumerable holding the random numbers (in memory for the CPU to use).</returns>
        public IEnumerable<float> GenerateUniformDistribution(int amount_of_numbers)
        {
            var result = new float[amount_of_numbers];
            UniformRand(DeviceId, amount_of_numbers, result);
            return result;
        }

        [DllImport("CudaSharperLibrary.dll")]
        private static extern void UniformRandDouble(int device_id, int amount_of_numbers, double[] result);

        /// <summary>
        /// Generate random numbers using XORWOW and a uniform distribution. This method utilizes the double-precision (FP64) capabilities of the GPU; this will perform worse than 
        /// using the single-precision (FP32) capabilities, and much worse on GeForce versus Quadro and Tesla. (Recommend only using this if you know the FP64 performance
        /// of the GPU being used).
        /// </summary>
        /// <param name="amount_of_numbers">The amount of random numbers to generate.</param>
        /// <param name="result">The array that will hold the random numbers (in memory for the CPU to use).</param>
        public void GenerateUniformDistributionDP(int amount_of_numbers, double[] result)
        {
            UniformRandDouble(DeviceId, amount_of_numbers, result);
        }

        /// <summary>
        /// Generate random numbers using XORWOW and a uniform distribution. This method utilizes the double-precision (FP64) capabilities of the GPU; this will perform worse than 
        /// using the single-precision (FP32) capabilities, and much worse on GeForce versus Quadro and Tesla. (Recommend only using this if you know the FP64 performance
        /// of the GPU being used).
        /// </summary>
        /// <param name="amount_of_numbers">The amount of random numbers to generate.</param>
        /// <returns>An IEnumerable holding the random numbers (in memory for the CPU to use).</returns>
        public IEnumerable<double> GenerateUniformDistributionDP(int amount_of_numbers)
        {
            var result = new double[amount_of_numbers];
            UniformRandDouble(DeviceId, amount_of_numbers, result);
            return result;
        }

        [DllImport("CudaSharperLibrary.dll")]
        private static extern void LogNormalRand(int device_id, int amount_of_numbers, float[] result, float mean, float stddev);

        /// <summary>
        /// Generates random numbers using XORWOW and a log normal distribution. This method utilizies the single-precision (FP32) capabilities of the GPU. If you need higher precision,
        /// there is a double-precision (FP64) version available; however, performance will be much worse, depending on the GPU being used.
        /// </summary>
        /// <param name="amount_of_numbers">The amount of random numbers to generate.</param>
        /// <param name="result">The array that will hold the random numbers (in memory for the CPU to use).</param>
        public void GenerateLogNormalDistribution(int amount_of_numbers, float[] result, float mean, float stddev)
        {
            LogNormalRand(DeviceId, amount_of_numbers, result, mean, stddev);
        }

        /// <summary>
        /// Generates random numbers using XORWOW and a log normal distribution. This method utilizies the single-precision (FP32) capabilities of the GPU. If you need higher precision,
        /// there is a double-precision (FP64) version available; however, performance will be much worse, depending on the GPU being used.
        /// </summary>
        /// <param name="amount_of_numbers">The amount of random numbers to generate.</param>
        /// <returns>An IEnumerable holding the random numbers (in memory for the CPU to use).</returns>
        public IEnumerable<float> GenerateLogNormalDistribution(int amount_of_numbers, float mean, float stddev)
        {
            var result = new float[amount_of_numbers];
            LogNormalRand(DeviceId, amount_of_numbers, result, mean, stddev);
            return result;
        }

        [DllImport("CudaSharperLibrary.dll")]
        private static extern void LogNormalRandDouble(int device_id, int amount_of_numbers, double[] result, float mean, float stddev);

        /// <summary>
        /// Generate random numbers using XORWOW and a log normal distribution. This method utilizes the double-precision (FP64) capabilities of the GPU; this will perform worse than 
        /// using the single-precision (FP32) capabilities, and much worse on GeForce versus Quadro and Tesla. (Recommend only using this if you know the FP64 performance
        /// of the GPU being used).
        /// </summary>
        /// <param name="amount_of_numbers">The amount of random numbers to generate.</param>
        /// <param name="result">The array that will hold the random numbers (in memory for the CPU to use).</param>
        /// <param name="mean">The mean (average) of the distribution.</param>
        /// <param name="stddev">The standard deviation of the distribution.</param>
        public void GenerateLogNormalDistributionDP(int amount_of_numbers, double[] result, float mean, float stddev)
        {
            LogNormalRandDouble(DeviceId, amount_of_numbers, result, mean, stddev);
        }

        /// <summary>
        /// Generate random numbers using XORWOW and a log normal distribution. This method utilizes the double-precision (FP64) capabilities of the GPU; this will perform worse than 
        /// using the single-precision (FP32) capabilities, and much worse on GeForce versus Quadro and Tesla. (Recommend only using this if you know the FP64 performance
        /// of the GPU being used).
        /// </summary>
        /// <param name="amount_of_numbers">The amount of random numbers to generate.</param>
        /// <param name="mean">The mean (average) of the distribution.</param>
        /// <param name="stddev">The standard deviation of the distribution.</param>
        /// <returns>An IEnumerable holding the random numbers (in memory for the CPU to use).</returns>
        public IEnumerable<double> GenerateLogNormalDistributionDP(int amount_of_numbers, float mean, float stddev)
        {
            var result = new double[amount_of_numbers];
            LogNormalRandDouble(DeviceId, amount_of_numbers, result, mean, stddev);
            return result;
        }

        [DllImport("CudaSharperLibrary.dll")]
        private static extern void NormalRand(int device_id, int amount_of_numbers, float[] result);

        /// <summary>
        /// Generates random numbers using XORWOW and a normal distribution. This method utilizies the single-precision (FP32) capabilities of the GPU. If you need higher precision,
        /// there is a double-precision (FP64) version available; however, performance will be much worse, depending on the GPU being used.
        /// </summary>
        /// <param name="amount_of_numbers">The amount of random numbers to generate.</param>
        /// <param name="result">The array that will hold the random numbers (in memory for the CPU to use).</param>
        public void GenerateNormalDistribution(int amount_of_numbers, float[] result)
        {
            NormalRand(DeviceId, amount_of_numbers, result);
        }

        /// <summary>
        /// Generates random numbers using XORWOW and a normal distribution. This method utilizies the single-precision (FP32) capabilities of the GPU. If you need higher precision,
        /// there is a double-precision (FP64) version available; however, performance will be much worse, depending on the GPU being used.
        /// </summary>
        /// <param name="amount_of_numbers">The amount of random numbers to generate.</param>
        /// <returns>An IEnumerable holding the random numbers (in memory for the CPU to use).</returns>
        public IEnumerable<float> GenerateNormalDistribution(int amount_of_numbers)
        {
            var result = new float[amount_of_numbers];
            NormalRand(DeviceId, amount_of_numbers, result);
            return result;
        }

        [DllImport("CudaSharperLibrary.dll")]
        private static extern void NormalRandDouble(int device_id, int amount_of_numbers, double[] result);

        /// <summary>
        /// Generates random numbers using XORWOW and a normal distribution. This method utilizes the double-precision (FP64) capabilities of the GPU; this will perform worse than 
        /// using the single-precision (FP32) capabilities, and much worse on GeForce versus Quadro and Tesla. (Recommend only using this if you know the FP64 performance
        /// of the GPU being used).
        /// </summary>
        /// <param name="amount_of_numbers">The amount of random numbers to generate.</param>
        /// <param name="result">An IEnumerable holding the random numbers (in memory for the CPU to use).</param>
        public void GenerateNormalDistributionDP(int amount_of_numbers, double[] result)
        {
            NormalRandDouble(DeviceId, amount_of_numbers, result);
        }

        /// <summary>
        /// Generates random numbers using XORWOW and a normal distribution. This method utilizes the double-precision (FP64) capabilities of the GPU; this will perform worse than 
        /// using the single-precision (FP32) capabilities, and much worse on GeForce versus Quadro and Tesla. (Recommend only using this if you know the FP64 performance
        /// of the GPU being used).
        /// </summary>
        /// <param name="amount_of_numbers">The amount of random numbers to generate.</param>
        /// <returns>An IEnumerable holding the random numbers (in memory for the CPU to use).</returns>
        public IEnumerable<double> GenerateNormalDistributionDP(int amount_of_numbers)
        {
            var result = new double[amount_of_numbers];
            NormalRandDouble(DeviceId, amount_of_numbers, result);
            return result;
        }

        [DllImport("CudaSharperLibrary.dll")]
        private static extern void PoissonRand(int device_id, int amount_of_numbers, int[] result, double lambda);

        public void GeneratePoissonDistribution(int amount_of_numbers, int[] result, double lambda)
        {
            PoissonRand(DeviceId, amount_of_numbers, result, lambda);
        }

        public IEnumerable<int> GeneratePoissonDistribution(int amount_of_numbers, double lambda)
        {
            var result = new int[amount_of_numbers];
            PoissonRand(DeviceId, amount_of_numbers, result, lambda);
            return result;
        }
    }
}
