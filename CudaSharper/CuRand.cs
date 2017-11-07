using System;

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
    public sealed class CuRand : IDisposable
    {
        private ICudaDevice CudaDeviceComponent { get; }
        private IntPtr PtrToUnmanagedClass { get; set; }

        public int DeviceId => CudaDeviceComponent.DeviceId;
        public long Max => CudaDeviceComponent.AllocationSize;

        static CuRand()
        {
            CudaSettings.Load();
        }

        public CuRand(ICudaDevice device)
        {
            CudaDeviceComponent = new CudaDevice(device.DeviceId, device.AllocationSize);
            PtrToUnmanagedClass = SafeNativeMethods.CreateRandomClass(CudaDeviceComponent.DeviceId, CudaDeviceComponent.AllocationSize);
        }
        
        public ICudaResult<float[]> GenerateUniformDistribution(long amount_of_numbers, float[] result)
        {
            var error = SafeNativeMethods.UniformRand(PtrToUnmanagedClass, result, amount_of_numbers);
            return new CudaResult<float[]>(error, result);
        }

        /// <summary>
        /// Generates random numbers using XORWOW and a uniform distribution. This method utilizies the single-precision (FP32) capabilities of the GPU. If you need higher precision,
        /// there is a double-precision (FP64) version available; however, performance will be much worse, depending on the GPU being used.
        /// </summary>
        /// <param name="amount_of_numbers">The amount of random numbers to generate.</param>
        /// <returns>An IEnumerable holding the random numbers (in memory for the CPU to use).</returns>
        public ICudaResult<float[]> GenerateUniformDistribution(long amount_of_numbers)
        {
            var result = new float[amount_of_numbers];
            return GenerateUniformDistribution(amount_of_numbers, result);
        }
        
        public ICudaResult<double[]> GenerateUniformDistributionDP(long amount_of_numbers, double[] result)
        {
            var error = SafeNativeMethods.UniformRandDouble(PtrToUnmanagedClass, result, amount_of_numbers);
            return new CudaResult<double[]>(error, result);
        }

        /// <summary>
        /// Generate random numbers using XORWOW and a uniform distribution. This method utilizes the double-precision (FP64) capabilities of the GPU; this will perform worse than 
        /// using the single-precision (FP32) capabilities, and much worse on GeForce versus Quadro and Tesla. (Recommend only using this if you know the FP64 performance
        /// of the GPU being used).
        /// </summary>
        /// <param name="amount_of_numbers">The amount of random numbers to generate.</param>
        /// <returns>An IEnumerable holding the random numbers (in memory for the CPU to use).</returns>
        public ICudaResult<double[]> GenerateUniformDistributionDP(long amount_of_numbers)
        {
            var result = new double[amount_of_numbers];
            return GenerateUniformDistributionDP(amount_of_numbers, result);
        }

        public ICudaResult<float[]> GenerateLogNormalDistribution(long amount_of_numbers, float[] result, float mean, float stddev)
        {
            var error = SafeNativeMethods.LogNormalRand(PtrToUnmanagedClass, result, amount_of_numbers, mean, stddev);
            return new CudaResult<float[]>(error, result);
        }

        /// <summary>
        /// Generates random numbers using XORWOW and a log normal distribution. This method utilizies the single-precision (FP32) capabilities of the GPU. If you need higher precision,
        /// there is a double-precision (FP64) version available; however, performance will be much worse, depending on the GPU being used.
        /// </summary>
        /// <param name="amount_of_numbers">The amount of random numbers to generate.</param>
        /// <returns>An IEnumerable holding the random numbers (in memory for the CPU to use).</returns>
        public ICudaResult<float[]> GenerateLogNormalDistribution(long amount_of_numbers, float mean, float stddev)
        {
            var result = new float[amount_of_numbers];
            return GenerateLogNormalDistribution(amount_of_numbers, result, mean, stddev);
        }

        public ICudaResult<double[]> GenerateLogNormalDistributionDP(long amount_of_numbers, double[] result, float mean, float stddev)
        {
            var error = SafeNativeMethods.LogNormalRandDouble(PtrToUnmanagedClass, result, amount_of_numbers, mean, stddev);
            return new CudaResult<double[]>(error, result);
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
        public ICudaResult<double[]> GenerateLogNormalDistributionDP(long amount_of_numbers, float mean, float stddev)
        {
            var result = new double[amount_of_numbers];
            return GenerateLogNormalDistributionDP(amount_of_numbers, result, mean, stddev);
        }

        public ICudaResult<float[]> GenerateNormalDistribution(long amount_of_numbers, float[] result)
        {
            var error = SafeNativeMethods.NormalRand(PtrToUnmanagedClass, result, amount_of_numbers);
            return new CudaResult<float[]>(error, result);
        }

        /// <summary>
        /// Generates random numbers using XORWOW and a normal distribution. This method utilizies the single-precision (FP32) capabilities of the GPU. If you need higher precision,
        /// there is a double-precision (FP64) version available; however, performance will be much worse, depending on the GPU being used.
        /// </summary>
        /// <param name="amount_of_numbers">The amount of random numbers to generate.</param>
        /// <returns>An IEnumerable holding the random numbers (in memory for the CPU to use).</returns>
        public ICudaResult<float[]> GenerateNormalDistribution(long amount_of_numbers)
        {
            var result = new float[amount_of_numbers];
            return GenerateNormalDistribution(amount_of_numbers, result);
        }

        public ICudaResult<double[]> GenerateNormalDistributionDP(long amount_of_numbers, double[] result)
        {
            var error = SafeNativeMethods.NormalRandDouble(PtrToUnmanagedClass, result, amount_of_numbers);
            return new CudaResult<double[]>(error, result);
        }

        /// <summary>
        /// Generates random numbers using XORWOW and a normal distribution. This method utilizes the double-precision (FP64) capabilities of the GPU; this will perform worse than 
        /// using the single-precision (FP32) capabilities, and much worse on GeForce versus Quadro and Tesla. (Recommend only using this if you know the FP64 performance
        /// of the GPU being used).
        /// </summary>
        /// <param name="amount_of_numbers">The amount of random numbers to generate.</param>
        /// <returns>An IEnumerable holding the random numbers (in memory for the CPU to use).</returns>
        public ICudaResult<double[]> GenerateNormalDistributionDP(long amount_of_numbers)
        {
            var result = new double[amount_of_numbers];
            return GenerateNormalDistributionDP(amount_of_numbers, result);
        }

        public ICudaResult<int[]> GeneratePoissonDistribution(long amount_of_numbers, int[] result, double lambda)
        {
            var error = SafeNativeMethods.PoissonRand(PtrToUnmanagedClass, result, amount_of_numbers, lambda);
            return new CudaResult<int[]>(error, result);
        }

        public ICudaResult<int[]> GeneratePoissonDistribution(long amount_of_numbers, double lambda)
        {
            var result = new int[amount_of_numbers];
            return GeneratePoissonDistribution(amount_of_numbers, result, lambda);
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
                    SafeNativeMethods.DisposeRandomClass(PtrToUnmanagedClass);
                    PtrToUnmanagedClass = IntPtr.Zero;
                }

                disposedValue = true;
            }
        }

        // TODO: override a finalizer only if Dispose(bool disposing) above has code to free unmanaged resources.
        ~CuRand()
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
