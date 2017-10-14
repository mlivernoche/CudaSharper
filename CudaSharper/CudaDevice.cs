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
    public class CudaDevice : ICudaDevice
    {
        public int DeviceId { get; }

        public CudaDevice(int device_id)
        {
            CudaSettings.Load();

            if ((device_id + 1) > CudaSettings.CudaDeviceCount)
            {
                throw new ArgumentOutOfRangeException("Bad DeviceId provided: not enough CUDA-enabled devices available. Devices available: " + CudaSettings.CudaDeviceCount + ". DeviceId: " + device_id);
            }

            DeviceId = device_id;
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
