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

        public int DeviceId => CudaDeviceComponent.DeviceId;

        public CuStats(int device_id)
        {
            CudaDeviceComponent = new CudaDevice(device_id);
        }

        public string GetCudaDeviceName()
        {
            return CudaDeviceComponent.GetCudaDeviceName();
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

        [DllImport("CudaSharperLibrary.dll")]
        private static extern double StandardDeviationFloat(uint device_id, float[] sample, ulong sample_size, double mean);

        public double StandardDeviation(float[] sample, double mean)
        {
            return StandardDeviationFloat((uint)CudaDeviceComponent.DeviceId, sample, (ulong)sample.Length, mean);
        }

        public double StandardDeviation(float[] sample)
        {
            return StandardDeviation(sample, sample.Average());
        }

        [DllImport("CudaSharperLibrary.dll")]
        private static extern double StandardDeviationDouble(uint device_id, double[] sample, ulong sample_size, double mean);

        public double StandardDeviation(double[] sample, double mean)
        {
            return StandardDeviationDouble((uint)CudaDeviceComponent.DeviceId, sample, (ulong)sample.Length, mean);
        }

        public double StandardDeviation(double[] sample)
        {
            return StandardDeviation(sample, sample.Average());
        }

        [DllImport("CudaSharperLibrary.dll")]
        private static extern double SampleCovarianceDouble(uint device_id, double[] x_array, double x_mean, double[] y_array, double y_mean, ulong sample_size);

        public double SampleCovariance(double[] x_array, double x_mean, double[] y_array, double y_mean)
        {
            return SampleCovarianceDouble((uint)CudaDeviceComponent.DeviceId, x_array, x_array.Average(), y_array, y_array.Average(), (uint) x_array.Length);
        }

        public double SampleCovariance(double[] x_array, double[] y_array)
        {
            return SampleCovariance(x_array, x_array.Average(), y_array, y_array.Average());
        }

        [DllImport("CudaSharperLibrary.dll")]
        private static extern double SampleCovarianceFloat(uint device_id, float[] x_array, double x_mean, float[] y_array, double y_mean, ulong sample_size);

        public double SampleCovariance(float[] x_array, double x_mean, float[] y_array, double y_mean)
        {
            return SampleCovarianceFloat((uint)CudaDeviceComponent.DeviceId, x_array, x_array.Average(), y_array, y_array.Average(), (uint)x_array.Length);
        }

        public double SampleCovariance(float[] x_array, float[] y_array)
        {
            return SampleCovariance(x_array, x_array.Average(), y_array, y_array.Average());
        }

        [DllImport("CudaSharperLibrary.dll")]
        private static extern double PearsonCorrelationFloat(uint device_id, float[] x_array, double x_mean, float[] y_array, double y_mean, ulong sample_size);

        public double Correlation(float[] x_array, double x_mean, float[] y_array, double y_mean)
        {
            return PearsonCorrelationFloat((uint)CudaDeviceComponent.DeviceId, x_array, x_array.Average(), y_array, y_array.Average(), (uint)x_array.Length);
        }

        public double Correlation(float[] x_array, float[] y_array)
        {
            return Correlation(x_array, x_array.Average(), y_array, y_array.Average());
        }

        [DllImport("CudaSharperLibrary.dll")]
        private static extern double PearsonCorrelationDouble(uint device_id, double[] x_array, double x_mean, double[] y_array, double y_mean, ulong sample_size);

        public double Correlation(double[] x_array, double x_mean, double[] y_array, double y_mean)
        {
            return PearsonCorrelationDouble((uint)CudaDeviceComponent.DeviceId, x_array, x_array.Average(), y_array, y_array.Average(), (uint)x_array.Length);
        }

        public double Correlation(double[] x_array, double[] y_array)
        {
            return Correlation(x_array, x_array.Average(), y_array, y_array.Average());
        }
    }
}
