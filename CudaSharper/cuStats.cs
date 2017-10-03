using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace CudaSharper
{
    internal static partial class SafeNativeMethods
    {
    }

    public class CuStats : ICudaDevice
    {
        private ICudaDevice CudaDeviceComponent { get; }

        public int DeviceId => CudaDeviceComponent.DeviceId;

        static CuStats()
        {
            CudaSettings.Load();
        }

        public CuStats(int device_id)
        {
            CudaDeviceComponent = new CudaDevice(device_id);
        }

        public CuStats(CudaDevice device)
        {
            CudaDeviceComponent = device;
        }

        public string GetCudaDeviceName()
        {
            return CudaDeviceComponent.GetCudaDeviceName();
        }

        public double SampleStandardDeviation(float[] sample, double mean)
        {
            return SafeNativeMethods.SampleStandardDeviationFloat((uint)CudaDeviceComponent.DeviceId, sample, (ulong)sample.LongLength, mean);
        }

        public double SampleStandardDeviation(float[] sample)
        {
            return SampleStandardDeviation(sample, sample.Average());
        }

        public double SampleStandardDeviation(double[] sample, double mean)
        {
            return SafeNativeMethods.SampleStandardDeviationDouble((uint)CudaDeviceComponent.DeviceId, sample, (ulong)sample.LongLength, mean);
        }

        public double SampleStandardDeviation(double[] sample)
        {
            return SampleStandardDeviation(sample, sample.Average());
        }

        public double StandardDeviation(float[] sample, double mean)
        {
            return SafeNativeMethods.StandardDeviationFloat((uint)CudaDeviceComponent.DeviceId, sample, (ulong)sample.LongLength, mean);
        }

        public double StandardDeviation(float[] sample)
        {
            return StandardDeviation(sample, sample.Average());
        }

        public double StandardDeviation(double[] sample, double mean)
        {
            return SafeNativeMethods.StandardDeviationDouble((uint)CudaDeviceComponent.DeviceId, sample, (ulong)sample.LongLength, mean);
        }

        public double StandardDeviation(double[] sample)
        {
            return StandardDeviation(sample, sample.Average());
        }

        public double SampleCovariance(float[] x_array, double x_mean, float[] y_array, double y_mean)
        {
            return SafeNativeMethods.SampleCovarianceFloat((uint)CudaDeviceComponent.DeviceId, x_array, x_array.Average(), y_array, y_array.Average(), (ulong)x_array.LongLength);
        }

        public double SampleCovariance(float[] x_array, float[] y_array)
        {
            return SampleCovariance(x_array, x_array.Average(), y_array, y_array.Average());
        }

        public double SampleCovariance(double[] x_array, double x_mean, double[] y_array, double y_mean)
        {
            return SafeNativeMethods.SampleCovarianceDouble((uint)CudaDeviceComponent.DeviceId, x_array, x_array.Average(), y_array, y_array.Average(), (ulong)x_array.LongLength);
        }

        public double SampleCovariance(double[] x_array, double[] y_array)
        {
            return SampleCovariance(x_array, x_array.Average(), y_array, y_array.Average());
        }

        public double Covariance(double[] x_array, double x_mean, double[] y_array, double y_mean)
        {
            return SafeNativeMethods.CovarianceDouble((uint)CudaDeviceComponent.DeviceId, x_array, x_array.Average(), y_array, y_array.Average(), (ulong)x_array.LongLength);
        }

        public double Covariance(double[] x_array, double[] y_array)
        {
            return Covariance(x_array, x_array.Average(), y_array, y_array.Average());
        }

        public double Covariance(float[] x_array, double x_mean, float[] y_array, double y_mean)
        {
            return SafeNativeMethods.CovarianceFloat((uint)CudaDeviceComponent.DeviceId, x_array, x_array.Average(), y_array, y_array.Average(), (ulong)x_array.LongLength);
        }

        public double Covariance(float[] x_array, float[] y_array)
        {
            return Covariance(x_array, x_array.Average(), y_array, y_array.Average());
        }

        public double Correlation(float[] x_array, double x_mean, float[] y_array, double y_mean)
        {
            return SafeNativeMethods.PearsonCorrelationFloat((uint)CudaDeviceComponent.DeviceId, x_array, x_array.Average(), y_array, y_array.Average(), (ulong)x_array.LongLength);
        }

        public double Correlation(float[] x_array, float[] y_array)
        {
            return Correlation(x_array, x_array.Average(), y_array, y_array.Average());
        }

        public double Correlation(double[] x_array, double x_mean, double[] y_array, double y_mean)
        {
            return SafeNativeMethods.PearsonCorrelationDouble((uint)CudaDeviceComponent.DeviceId, x_array, x_array.Average(), y_array, y_array.Average(), (ulong)x_array.LongLength);
        }

        public double Correlation(double[] x_array, double[] y_array)
        {
            return Correlation(x_array, x_array.Average(), y_array, y_array.Average());
        }

        public double[][] CorrelationMatrix(double[][] sets_of_scalars)
        {
            var set_length = sets_of_scalars.Length;
            var C = new double[set_length][];

            for (int i = 0; i < set_length; i++)
            {
                C[i] = new double[set_length];

                for (int j = 0; j < set_length; j++)
                {
                    C[i][j] = Correlation(sets_of_scalars[i], sets_of_scalars[j]);
                }
            }

            return C;
        }

        public float[][] CorrelationMatrix(float[][] sets_of_scalars)
        {
            var set_length = sets_of_scalars.LongLength;
            var C = new float[set_length][];

            for (long i = 0; i < set_length; i++)
            {
                C[i] = new float[set_length];

                for (long j = 0; j < set_length; j++)
                {
                    C[i][j] = (float)Correlation(sets_of_scalars[i], sets_of_scalars[j]);
                }
            }

            return C;
        }

        public double[][] CovarianceMatrix(double[][] sets_of_scalars)
        {
            var set_length = sets_of_scalars.LongLength;
            var C = new double[set_length][];

            for (long i = 0; i < set_length; i++)
            {
                C[i] = new double[set_length];

                for (long j = 0; j < set_length; j++)
                {
                    C[i][j] = Covariance(sets_of_scalars[i], sets_of_scalars[j]);
                }
            }

            return C;
        }

        public float[][] CovarianceMatrix(float[][] sets_of_scalars)
        {
            var set_length = sets_of_scalars.LongLength;
            var C = new float[set_length][];

            for (long i = 0; i < set_length; i++)
            {
                C[i] = new float[set_length];

                for (long j = 0; j < set_length; j++)
                {
                    C[i][j] = (float)Covariance(sets_of_scalars[i], sets_of_scalars[j]);
                }
            }

            return C;
        }

        public double VaR(float[] invested_amounts, float[][] covariance_matrix, double confidence_level, int time_period)
        {
            var cuArray = new CuArray(DeviceId);

            var invested_amounts_horizontal = new float[][] { invested_amounts };
            var covariance_times_beta_horizontal = cuArray.Multiply(
                CUBLAS_OP.DO_NOT_TRANSPOSE, CUBLAS_OP.DO_NOT_TRANSPOSE,
                1,
                invested_amounts_horizontal,
                covariance_matrix,
                0);

            var covariance_times_beta_vertical = cuArray.Multiply(
                CUBLAS_OP.TRANSPOSE, CUBLAS_OP.DO_NOT_TRANSPOSE,
                1,
                covariance_times_beta_horizontal,
                invested_amounts_horizontal,
                0);

            if(covariance_times_beta_vertical.Length > 1 || covariance_times_beta_vertical[0].Length > 1)
            {
                throw new ArgumentOutOfRangeException("The matrix given for Beta * CovMatrix * Beta^T was bigger than one.");
            }

            return Math.Sqrt(covariance_times_beta_vertical[0][0]) * confidence_level * Math.Sqrt(time_period);
        }
    }
}
