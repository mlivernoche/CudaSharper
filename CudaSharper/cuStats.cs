using System;
using System.Linq;

namespace CudaSharper
{
    public sealed class CuStats : IDisposable
    {
        private ICudaDevice CudaDeviceComponent { get; }
        private IntPtr PtrToUnmanagedClass { get; set; }

        public int DeviceId => CudaDeviceComponent.DeviceId;

        static CuStats()
        {
            CudaSettings.Load();
        }

        public CuStats(ICudaDevice device)
        {
            CudaDeviceComponent = new CudaDevice(device.DeviceId, device.AllocationSize);
            PtrToUnmanagedClass = SafeNativeMethods.CreateStatClass(CudaDeviceComponent.DeviceId, CudaDeviceComponent.AllocationSize);
        }

        public ICudaResult<double> SampleStandardDeviation(float[] sample, float mean)
        {
            double result = 0;
            var error_code = SafeNativeMethods.SampleStandardDeviationFloat(
                PtrToUnmanagedClass,
                ref result,
                sample, sample.LongLength, mean);
            return new CudaResult<double>(error_code, result);
        }

        public ICudaResult<double> SampleStandardDeviation(float[] sample)
        {
            return SampleStandardDeviation(sample, sample.Average());
        }

        public ICudaResult<double> SampleStandardDeviation(double[] sample, double mean)
        {
            double result = 0;
            var error_code = SafeNativeMethods.SampleStandardDeviationDouble(
                PtrToUnmanagedClass,
                ref result,
                sample, sample.LongLength, mean);
            return new CudaResult<double>(error_code, result);
        }

        public ICudaResult<double> SampleStandardDeviation(double[] sample)
        {
            return SampleStandardDeviation(sample, sample.Average());
        }

        public ICudaResult<double> StandardDeviation(float[] sample, float mean)
        {
            if (sample.LongLength > CudaDeviceComponent.AllocationSize) throw new ArgumentOutOfRangeException("Array bigger than allocation size.");

            double result = 0;
            var error_code = SafeNativeMethods.StandardDeviationFloat(
                PtrToUnmanagedClass,
                ref result,
                sample, sample.LongLength, mean);
            return new CudaResult<double>(error_code, result);
        }

        public ICudaResult<double> StandardDeviation(float[] sample)
        {
            return StandardDeviation(sample, sample.Average());
        }

        public ICudaResult<double> StandardDeviation(double[] sample, double mean)
        {
            if (sample.LongLength > CudaDeviceComponent.AllocationSize) throw new ArgumentOutOfRangeException("Array bigger than allocation size.");

            double result = 0;
            var error_code = SafeNativeMethods.StandardDeviationDouble(
                PtrToUnmanagedClass,
                ref result,
                sample, sample.LongLength, mean);
            return new CudaResult<double>(error_code, result);
        }

        public ICudaResult<double> StandardDeviation(double[] sample)
        {
            return StandardDeviation(sample, sample.Average());
        }

        public ICudaResult<double> Variance(float[] array, float mean)
        {
            var std = StandardDeviation(array, mean);
            return new CudaResult<double>(std.Error, Math.Pow(std.Result, 2));
        }

        public ICudaResult<double> Variance(float[] array)
        {
            var std = StandardDeviation(array, array.Average());
            return new CudaResult<double>(std.Error, Math.Pow(std.Result, 2));
        }

        public ICudaResult<double> Variance(double[] array, double mean)
        {
            var std = StandardDeviation(array, mean);
            return new CudaResult<double>(std.Error, Math.Pow(std.Result, 2));
        }

        public ICudaResult<double> Variance(double[] array)
        {
            var std = StandardDeviation(array, array.Average());
            return new CudaResult<double>(std.Error, Math.Pow(std.Result, 2));
        }

        public ICudaResult<double> SampleVariance(float[] array, float mean)
        {
            var std = SampleStandardDeviation(array, mean);
            return new CudaResult<double>(std.Error, Math.Pow(std.Result, 2));
        }

        public ICudaResult<double> SampleVariance(float[] array)
        {
            var std = SampleStandardDeviation(array, array.Average());
            return new CudaResult<double>(std.Error, Math.Pow(std.Result, 2));
        }

        public ICudaResult<double> SampleVariance(double[] array, double mean)
        {
            var std = SampleStandardDeviation(array, mean);
            return new CudaResult<double>(std.Error, Math.Pow(std.Result, 2));
        }

        public ICudaResult<double> SampleVariance(double[] array)
        {
            var std = SampleStandardDeviation(array, array.Average());
            return new CudaResult<double>(std.Error, Math.Pow(std.Result, 2));
        }

        public ICudaResult<double> SampleCovariance(float[] x_array, float x_mean, float[] y_array, float y_mean)
        {
            double result = 0;
            var error_code = SafeNativeMethods.SampleCovarianceFloat(
                PtrToUnmanagedClass,
                ref result,
                x_array, x_mean,
                y_array, y_mean,
                x_array.LongLength);
            return new CudaResult<double>(error_code, result);
        }

        public ICudaResult<double> SampleCovariance(float[] x_array, float[] y_array)
        {
            return SampleCovariance(x_array, x_array.Average(), y_array, y_array.Average());
        }

        public ICudaResult<double> SampleCovariance(double[] x_array, double x_mean, double[] y_array, double y_mean)
        {
            double result = 0;
            var error_code = SafeNativeMethods.SampleCovarianceDouble(
                PtrToUnmanagedClass,
                ref result,
                x_array, x_mean,
                y_array, y_mean,
                x_array.LongLength);
            return new CudaResult<double>(error_code, result);
        }

        public ICudaResult<double> SampleCovariance(double[] x_array, double[] y_array)
        {
            return SampleCovariance(x_array, x_array.Average(), y_array, y_array.Average());
        }

        public ICudaResult<double> Covariance(double[] x_array, double x_mean, double[] y_array, double y_mean)
        {
            double result = 0;
            var error_code = SafeNativeMethods.CovarianceDouble(
                PtrToUnmanagedClass,
                ref result,
                x_array, x_mean,
                y_array, y_mean,
                x_array.LongLength);
            return new CudaResult<double>(error_code, result);
        }

        public ICudaResult<double> Covariance(float[] x_array, float x_mean, float[] y_array, float y_mean)
        {
            double result = 0;
            var error_code = SafeNativeMethods.CovarianceFloat(
                PtrToUnmanagedClass,
                ref result,
                x_array, x_mean,
                y_array, y_mean,
                x_array.LongLength);
            return new CudaResult<double>(error_code, result);
        }

        public ICudaResult<double> Covariance(double[] x_array, double[] y_array)
        {
            return Covariance(x_array, x_array.Average(), y_array, y_array.Average());
        }

        public ICudaResult<double> Covariance(float[] x_array, float[] y_array)
        {
            return Covariance(x_array, x_array.Average(), y_array, y_array.Average());
        }

        public ICudaResult<double> Correlation(float[] x_array, float x_mean, float[] y_array, float y_mean)
        {
            double result = 0;
            var error_code = SafeNativeMethods.PearsonCorrelationFloat(
                PtrToUnmanagedClass,
                ref result,
                x_array, x_mean,
                y_array, y_mean,
                x_array.LongLength);
            return new CudaResult<double>(error_code, result);
        }

        public ICudaResult<double> Correlation(float[] x_array, float[] y_array)
        {
            return Correlation(x_array, x_array.Average(), y_array, y_array.Average());
        }

        public ICudaResult<double> Correlation(double[] x_array, double x_mean, double[] y_array, double y_mean)
        {
            double result = 0;
            var error_code = SafeNativeMethods.PearsonCorrelationDouble(
                PtrToUnmanagedClass,
                ref result,
                x_array, x_mean,
                y_array, y_mean,
                x_array.LongLength);
            return new CudaResult<double>(error_code, result);
        }

        public ICudaResult<double> Correlation(double[] x_array, double[] y_array)
        {
            return Correlation(x_array, x_array.Average(), y_array, y_array.Average());
        }

        public ICudaResult<double> Autocorrelation(float[] x_array, float x_mean, float[] y_array, float y_mean, int lag)
        {
            if (lag < 0) throw new ArgumentOutOfRangeException($"Lag cannot be less than 0. Given: {lag}");

            var errorCode = CudaError.Success;

            var length = x_array.LongLength - lag;
            var array1 = new double[length];
            var array2 = new double[length];
            Array.Copy(x_array, 0, array1, 0, length);
            Array.Copy(y_array, lag, array2, 0, length);

            if (array1.Length != array2.Length) throw new AccessViolationException($"Array size mismatch: {array1.Length} vs {array2.Length}");
            if (array1.Length > CudaDeviceComponent.AllocationSize) throw new ArgumentOutOfRangeException($"Array bigger than allocation size: {array1.Length} vs {CudaDeviceComponent.AllocationSize}");

            var std = Variance(x_array, x_mean);
            if (std.Error != CudaError.Success) errorCode = std.Error;
            
            var cov = Covariance(array1, x_mean, array2, y_mean);
            if (cov.Error != CudaError.Success) errorCode = cov.Error;

            return new CudaResult<double>(errorCode, cov.Result / std.Result);
        }

        public ICudaResult<double> Autocorrelation(float[] x_array, float[] y_array, int lag)
        {
            return Autocorrelation(x_array, x_array.Average(), y_array, y_array.Average(), lag);
        }

        public ICudaResult<double> Autocorrelation(double[] x_array, double x_mean, double[] y_array, double y_mean, int lag)
        {
            if (lag < 0) throw new ArgumentOutOfRangeException($"Lag cannot be less than 0. Given: {lag}");

            var errorCode = CudaError.Success;
            
            var length = x_array.LongLength - lag;
            var array1 = new double[length];
            var array2 = new double[length];
            Array.Copy(x_array, 0, array1, 0, length);
            Array.Copy(y_array, lag, array2, 0, length);

            if (array1.Length != array2.Length) throw new AccessViolationException($"Array size mismatch: {array1.Length} vs {array2.Length}");
            if (array1.Length > CudaDeviceComponent.AllocationSize) throw new ArgumentOutOfRangeException($"Array bigger than allocation size: {array1.Length} vs {CudaDeviceComponent.AllocationSize}");

            var std = Variance(x_array, x_mean);
            if (std.Error != CudaError.Success) errorCode = std.Error;
            
            var cov = Covariance(array1, x_mean, array2, y_mean);
            if (cov.Error != CudaError.Success) errorCode = cov.Error;

            return new CudaResult<double>(errorCode, cov.Result / std.Result);
        }

        public ICudaResult<double> Autocorrelation(double[] x_array, double[] y_array, int lag)
        {
            return Autocorrelation(x_array, x_array.Average(), y_array, y_array.Average(), lag);
        }

        public ICudaResult<double[][]> CorrelationMatrix(float[][] sets_of_scalars)
        {
            var set_length = sets_of_scalars.LongLength;
            var C = new double[set_length][];
            var error_code = CudaError.Success;

            for (long i = 0; i < set_length; i++)
            {
                C[i] = new double[set_length];

                for (long j = 0; j < set_length; j++)
                {
                    // Correlation(x, y) will always return a double, but it will use FP32 if given
                    // floats or FP64 given doubles.
                    var corr = Correlation(sets_of_scalars[i], sets_of_scalars[j]);
                    error_code = corr.Error != CudaError.Success ? corr.Error : CudaError.Success;
                    C[i][j] = corr.Result;
                }
            }

            return new CudaResult<double[][]>(error_code, C);
        }

        public ICudaResult<double[][]> CorrelationMatrix(double[][] sets_of_scalars)
        {
            var set_length = sets_of_scalars.LongLength;
            var C = new double[set_length][];
            var error_code = CudaError.Success;

            for (long i = 0; i < set_length; i++)
            {
                C[i] = new double[set_length];

                for (long j = 0; j < set_length; j++)
                {
                    var corr = Correlation(sets_of_scalars[i], sets_of_scalars[j]);
                    error_code = corr.Error != CudaError.Success ? corr.Error : CudaError.Success;
                    C[i][j] = corr.Result;
                }
            }

            return new CudaResult<double[][]>(error_code, C);
        }

        public ICudaResult<double[][]> CovarianceMatrix(float[][] sets_of_scalars)
        {
            var set_length = sets_of_scalars.LongLength;
            var C = new double[set_length][];
            var error_code = CudaError.Success;

            for (long i = 0; i < set_length; i++)
            {
                C[i] = new double[set_length];

                for (long j = 0; j < set_length; j++)
                {
                    var cov = Covariance(sets_of_scalars[i], sets_of_scalars[j]);
                    error_code = cov.Error != CudaError.Success ? cov.Error : CudaError.Success;
                    C[i][j] = cov.Result;
                }
            }

            return new CudaResult<double[][]>(error_code, C);
        }

        public ICudaResult<double[][]> CovarianceMatrix(double[][] sets_of_scalars)
        {
            var set_length = sets_of_scalars.LongLength;
            var C = new double[set_length][];
            var error_code = CudaError.Success;

            for (long i = 0; i < set_length; i++)
            {
                C[i] = new double[set_length];

                for (long j = 0; j < set_length; j++)
                {
                    var cov = Covariance(sets_of_scalars[i], sets_of_scalars[j]);
                    error_code = cov.Error != CudaError.Success ? cov.Error : CudaError.Success;
                    C[i][j] = cov.Result;
                }
            }

            return new CudaResult<double[][]>(error_code, C);
        }

        /// <summary>
        /// Calculates the Value-at-Risk for a portfolio.
        /// </summary>
        /// <param name="invested_amounts">An array (1xN matrix) of the amounts invested in each portfolio.</param>
        /// <param name="covariance_matrix">A covariance matrix (NxN matrix) of the portfolio for the given time period.</param>
        /// <param name="confidence_level">The confidence level. This should be in units of standard deviation of a normal distribution (e.g., 0.90 = 1.645).</param>
        /// <param name="time_period">The time period for measuring risk.</param>
        /// <returns>The Value-at-Risk. No units involved.</returns>
        public double VaR(float[] invested_amounts, float[][] covariance_matrix, double confidence_level, int time_period)
        {
            using (var cuArray = new CuArray(new CudaDevice(CudaDeviceComponent.DeviceId, CudaDeviceComponent.AllocationSize)))
            {
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
                    covariance_times_beta_horizontal.Result,
                    invested_amounts_horizontal,
                    0);

                if (covariance_times_beta_vertical.Result.Length > 1 || covariance_times_beta_vertical.Result[0].Length > 1)
                {
                    throw new ArgumentOutOfRangeException("The matrix given for Beta * CovMatrix * Beta^T was bigger than one.");
                }

                return Math.Sqrt(covariance_times_beta_vertical.Result[0][0]) * confidence_level * Math.Sqrt(time_period);
            }
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
                    SafeNativeMethods.DisposeStatClass(PtrToUnmanagedClass);
                    PtrToUnmanagedClass = IntPtr.Zero;
                }

                disposedValue = true;
            }
        }

        // TODO: override a finalizer only if Dispose(bool disposing) above has code to free unmanaged resources.
        ~CuStats()
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
