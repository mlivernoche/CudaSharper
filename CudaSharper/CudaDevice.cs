using System;
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
    // This classes allows one to define a CUDA device, without allocating memory.
    // Importantly, this class can be passed around, but CudaDeviceHeavy cannot be passed around.

        /// <summary>
        /// This class defines the parameters for a CUDA device. A CUDA device is a software abstraction
        /// that represents a reuseble device context. In particular, this is injected into a seperate class
        /// called CudaDeviceHeavy, which calls into the C++ codebase. This creates cuda_device, cuda_mem_mngmt,
        /// and cuda_device_ptr, which all allow the reusing of allocated device memory throughout the life of the
        /// object.
        /// </summary>
    public class CudaDevice : ICudaDevice
    {
        public int DeviceId { get; }
        public long AllocationSize { get; }

        public CudaDevice(int device_id, long allocation)
        {
            DeviceId = device_id;
            AllocationSize = allocation;
        }

        public string GetCudaDeviceName()
        {
            var result = DTM.GetCudaDeviceName(DeviceId);

            if (result.Error != CudaError.Success)
                throw new Exception("Failed to get GPU name! Error provided: " + result.Error.ToString());

            return result.Result;
        }
    }
}
