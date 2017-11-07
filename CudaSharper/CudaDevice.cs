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
    /// <summary>
    /// This class defines the parameters for a CUDA device. In this context, a CUDA device is a software abstraction
    /// that represents a reuseble device context for a physical, CUDA-enabled hardware device. In particular,
    /// this allows the reuse of allocated device memory, thus eliminating the need for constant allocating and freeing.
    /// 
    /// In CSL, a cuda_device is a collection of pointers to individual cuda_device_ptr<T> objects, which each handle the allocation
    /// and freeing of its own device memory address. In its current implementation, cuda_device has four cuda_device_ptr objects:
    /// one for Int32, one for Int64, one for float, and one for double. In other words, a single cuda_device that specifies an
    /// allocation size of, say, 50,000, can have more than 50,000 bytes. At maximum, it can allocate on the device 50,000*sizeof(Int32) + 
    /// 50,000*sizeof(Int64) + 50,000*sizeof(float) + 50,000*sizeof(double). Or, 1,200,000 bytes (1.2MB). As noted earlier, however,
    /// these device memory allocations are "lazy," so 50,000 doubles will only be allocated if they are requested. If they are not requested,
    /// then they are not allocated.
    /// </summary>
    public class CudaDevice : ICudaDevice
    {
        public int DeviceId { get; }
        public long AllocationSize { get; }
        public string DeviceName => GetCudaDeviceName();

        /// <summary>
        /// CudaDevice is a "prototype" class that is passed to the CSL, which is used to build appropriate C++ objects.
        /// Namely, they specify the amount of memory that each C++ object should allocate for the CUDA device. These allocations
        /// are "lazy," in that they aren't done until they are needed (and if they aren't needed, then they are never allocated).
        /// Therefore, this class also specifies the maximum amount of memory that should be allocated. Objects in C# (like cuStats)
        /// do not have to use the entire memory space, but they cannot go above it. So, if you specify something like
        /// new CudaDevice(0, 100_000), then the C++ objects will allocate the appropriate memory (types included) for 100,000 "types"
        /// (e.g., 100,000 ints, 100,000 floats, etc. as they are needed). Any subsequent calls based on this information that attempt
        /// to access greater than 100,000 "types" will result in memory access violation exception. You do not need to use the entire
        /// max_allocation everytime, but you cannot use more than that at any time.
        /// </summary>
        /// <param name="device_id">The GPU device id. These start at 0 for the system default, 1 for the next, etc. Consult your system specs.</param>
        /// <param name="max_allocation">The amount of memory that can be allocated. CSL will only allocate memory as it is requested. Further, any calls to this device cannot reference more than this amount, but they can reference less than it.</param>
        public CudaDevice(int device_id, long max_allocation)
        {
            DeviceId = device_id;
            AllocationSize = max_allocation;
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
