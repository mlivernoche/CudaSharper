using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;

namespace CudaSharper
{
    public class CuStats : ICudaDevice
    {
        private ICudaDevice CudaDeviceComponent { get; }

        public int DeviceId => throw new NotImplementedException();

        public CuStats(int device_id)
        {
            CudaDeviceComponent = new CudaDevice(device_id);
        }

        public int CudaDevicesCount()
        {
            return CudaDeviceComponent.CudaDevicesCount();
        }

        public string GetCudaDeviceName(int device_id)
        {
            return CudaDeviceComponent.GetCudaDeviceName(device_id);
        }

        [DllImport("CudaSharperLibrary.dll")]
        private static extern double SampleStandardDeviationFloat(uint device_id, float[] sample, ulong sample_size, double mean);

        public double SampleStandardDeviation(float[] sample, double mean)
        {
            return SampleStandardDeviationFloat((uint)CudaDeviceComponent.DeviceId, sample, (ulong)sample.Length, mean);
        }

        public double SampleStandardDeviation(float[] sample)
        {
            return SampleStandardDeviation(sample, sample.Average());
        }

        [DllImport("CudaSharperLibrary.dll")]
        private static extern double SampleStandardDeviationDouble(uint device_id, double[] sample, ulong sample_size, double mean);

        public double SampleStandardDeviation(double[] sample, double mean)
        {
            return SampleStandardDeviationDouble((uint)CudaDeviceComponent.DeviceId, sample, (ulong)sample.Length, mean);
        }

        public double SampleStandardDeviation(double[] sample)
        {
            return SampleStandardDeviation(sample, sample.Average());
        }
    }
}
