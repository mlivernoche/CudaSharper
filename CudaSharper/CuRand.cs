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
    public class CuRand : ICudaDevice
    {
        private ICudaDevice CudaDeviceComponent { get; }

        public int DeviceId => CudaDeviceComponent.DeviceId;

        static CuRand()
        {
            CudaSettings.Load();
        }

        public CuRand(int device_id)
        {
            CudaDeviceComponent = new CudaDevice(device_id);
        }

        public CuRand(CudaDevice device)
        {
            CudaDeviceComponent = device;
        }

        public string GetCudaDeviceName()
        {
            return CudaDeviceComponent.GetCudaDeviceName();
        }

        /// <summary>
        /// Generates random numbers using XORWOW and a uniform distribution. This method utilizies the single-precision (FP32) capabilities of the GPU. If you need higher precision,
        /// there is a double-precision (FP64) version available; however, performance will be much worse, depending on the GPU being used.
        /// </summary>
        /// <param name="amount_of_numbers">The amount of random numbers to generate.</param>
        /// <returns>An IEnumerable holding the random numbers (in memory for the CPU to use).</returns>
        public ICudaResult<float[]> GenerateUniformDistribution(int amount_of_numbers)
        {
            var result = new float[amount_of_numbers];
            var error = SafeNativeMethods.UniformRand(DeviceId, result, amount_of_numbers);
            return new CudaResult<float[]>(error, result);
        }

        /// <summary>
        /// Generate random numbers using XORWOW and a uniform distribution. This method utilizes the double-precision (FP64) capabilities of the GPU; this will perform worse than 
        /// using the single-precision (FP32) capabilities, and much worse on GeForce versus Quadro and Tesla. (Recommend only using this if you know the FP64 performance
        /// of the GPU being used).
        /// </summary>
        /// <param name="amount_of_numbers">The amount of random numbers to generate.</param>
        /// <returns>An IEnumerable holding the random numbers (in memory for the CPU to use).</returns>
        public ICudaResult<double[]> GenerateUniformDistributionDP(int amount_of_numbers)
        {
            var result = new double[amount_of_numbers];
            var error = SafeNativeMethods.UniformRandDouble(DeviceId, result, amount_of_numbers);
            return new CudaResult<double[]>(error, result);
        }

        /// <summary>
        /// Generates random numbers using XORWOW and a log normal distribution. This method utilizies the single-precision (FP32) capabilities of the GPU. If you need higher precision,
        /// there is a double-precision (FP64) version available; however, performance will be much worse, depending on the GPU being used.
        /// </summary>
        /// <param name="amount_of_numbers">The amount of random numbers to generate.</param>
        /// <returns>An IEnumerable holding the random numbers (in memory for the CPU to use).</returns>
        public ICudaResult<float[]> GenerateLogNormalDistribution(int amount_of_numbers, float mean, float stddev)
        {
            var result = new float[amount_of_numbers];
            var error = SafeNativeMethods.LogNormalRand(DeviceId, result, amount_of_numbers, mean, stddev);
            return new CudaResult<float[]>(error, result);
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
        public ICudaResult<double[]> GenerateLogNormalDistributionDP(int amount_of_numbers, float mean, float stddev)
        {
            var result = new double[amount_of_numbers];
            var error = SafeNativeMethods.LogNormalRandDouble(DeviceId, result, amount_of_numbers, mean, stddev);
            return new CudaResult<double[]>(error, result);
        }

        /// <summary>
        /// Generates random numbers using XORWOW and a normal distribution. This method utilizies the single-precision (FP32) capabilities of the GPU. If you need higher precision,
        /// there is a double-precision (FP64) version available; however, performance will be much worse, depending on the GPU being used.
        /// </summary>
        /// <param name="amount_of_numbers">The amount of random numbers to generate.</param>
        /// <returns>An IEnumerable holding the random numbers (in memory for the CPU to use).</returns>
        public ICudaResult<float[]> GenerateNormalDistribution(int amount_of_numbers)
        {
            var result = new float[amount_of_numbers];
            var error = SafeNativeMethods.NormalRand(DeviceId, result, amount_of_numbers);
            return new CudaResult<float[]>(error, result);
        }

        /// <summary>
        /// Generates random numbers using XORWOW and a normal distribution. This method utilizes the double-precision (FP64) capabilities of the GPU; this will perform worse than 
        /// using the single-precision (FP32) capabilities, and much worse on GeForce versus Quadro and Tesla. (Recommend only using this if you know the FP64 performance
        /// of the GPU being used).
        /// </summary>
        /// <param name="amount_of_numbers">The amount of random numbers to generate.</param>
        /// <returns>An IEnumerable holding the random numbers (in memory for the CPU to use).</returns>
        public ICudaResult<double[]> GenerateNormalDistributionDP(int amount_of_numbers)
        {
            var result = new double[amount_of_numbers];
            var error = SafeNativeMethods.NormalRandDouble(DeviceId, result, amount_of_numbers);
            return new CudaResult<double[]>(error, result);
        }

        public ICudaResult<int[]> GeneratePoissonDistribution(int amount_of_numbers, double lambda)
        {
            var result = new int[amount_of_numbers];
            var error = SafeNativeMethods.PoissonRand(DeviceId, result, amount_of_numbers, lambda);
            return new CudaResult<int[]>(error, result);
        }
    }
}
