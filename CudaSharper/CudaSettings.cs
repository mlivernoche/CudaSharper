using System;
using System.Runtime.InteropServices;
using System.Threading;

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
    public class CudaSettings : IDisposable
    {
        private static bool WorkingDirSet { get; set; } = false;
        private static object LoadingLock { get; set; } = new object();

        public static int CudaDeviceCount { get; private set; } = 0;

        public static string Version => "v0.2.1";

        public static void Load(string working_directory)
        {
            lock (LoadingLock)
            {
                if (WorkingDirSet)
                    return;

                Environment.SetEnvironmentVariable("PATH", System.Environment.GetEnvironmentVariable("PATH") + working_directory, EnvironmentVariableTarget.Process);
                try
                {
                    CudaDeviceCount = SafeNativeMethods.GetCudaDeviceCount();
                    WorkingDirSet = true;
                }
                catch (DllNotFoundException e)
                {
                    Console.WriteLine(Environment.GetEnvironmentVariable("PATH"));
                    Console.WriteLine(e.Message);
                    throw e;
                }
            }
        }

        public static void Load()
        {
            Load(AppDomain.CurrentDomain.BaseDirectory);
        }

        #region IDisposable Support
        private bool disposedValue = false; // To detect redundant calls

        protected virtual void Dispose(bool disposing)
        {
            if (!disposedValue)
            {
                if (disposing)
                {
                    // TODO: dispose managed state (managed objects).
                }

                // TODO: free unmanaged resources (unmanaged objects) and override a finalizer below.
                // TODO: set large fields to null.

                DTM.ResetCudaDevice();
                //SafeNativeMethods.cuStats_Dispose();

                disposedValue = true;
            }
        }

        // TODO: override a finalizer only if Dispose(bool disposing) above has code to free unmanaged resources.
        ~CudaSettings()
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
