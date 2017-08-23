using System;
using System.Collections.Generic;
using System.Diagnostics;
using CudaSharper;

namespace CudaExample
{
    public static class Program
    {
        private static void PerformanceTimer(Action cuda, string func_name)
        {
            var timer = new Stopwatch();
            timer.Start();
            cuda();
            timer.Stop();
            Console.WriteLine(String.Format("{0} took {1} ms.", func_name, timer.ElapsedMilliseconds));
        }

        static void Main(string[] args)
        {
            CudaSettings.Load();
            var range = 100_000;

            for (int i = 0; i < 1000; i++)
            {
                var cuRand = new CuRand(1);

                Console.WriteLine("Executing CUDA kernels on a " + cuRand.GetCudaDeviceName(1));

                IEnumerable<float> uniform_rand;
                IEnumerable<double> uniform_rand_double;
                PerformanceTimer(() => uniform_rand = cuRand.GenerateUniformDistribution(range), nameof(cuRand.GenerateUniformDistribution));
                PerformanceTimer(() => uniform_rand_double = cuRand.GenerateUniformDistributionDP(range), nameof(cuRand.GenerateUniformDistributionDP));

                IEnumerable<float> normal_rand;
                IEnumerable<double> normal_rand_double;
                PerformanceTimer(() => normal_rand = cuRand.GenerateNormalDistribution(range), nameof(cuRand.GenerateNormalDistribution));
                PerformanceTimer(() => normal_rand_double = cuRand.GenerateNormalDistributionDP(range), nameof(cuRand.GenerateNormalDistributionDP));

                IEnumerable<float> log_normal_rand;
                IEnumerable<double> log_normal_rand_double;
                PerformanceTimer(() => log_normal_rand = cuRand.GenerateLogNormalDistribution(range, 5, 1), nameof(cuRand.GenerateLogNormalDistribution));
                PerformanceTimer(() => log_normal_rand_double = cuRand.GenerateLogNormalDistributionDP(range, 5, 1), nameof(cuRand.GenerateNormalDistributionDP));

                IEnumerable<int> poisson_rand;
                PerformanceTimer(() => poisson_rand = cuRand.GeneratePoissonDistribution(range, 3), nameof(cuRand.GeneratePoissonDistribution));

                Console.ReadKey();
            }
        }
    }
}
