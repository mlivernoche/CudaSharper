using System.Collections.Generic;

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
    internal interface ICuda : ICudaDevice
    {
        /*
         * SplitArray:
         * - int[], (IEnumerable<int>, IEnumerable<int>)
         * - long[], (IEnumerable<float>, IEnumerable<float>)
         * - float[], (IEnumerable<float>, IEnumerable<float>)
         * - double[], (IEnumerable<double>, IEnumerable<double>)
         */
        void SplitArray(int[] src, int[] array1, int[] array2, uint length, uint split_index);
        (IEnumerable<int>, IEnumerable<int>) SplitArray(int[] src, uint length, uint split_index);

        void SplitArray(long[] src, long[] array1, long[] array2, uint length, uint split_index);
        (IEnumerable<long>, IEnumerable<long>) SplitArray(long[] src, uint length, uint split_index);

        void SplitArray(float[] src, float[] array1, float[] array2, uint length, uint split_index);
        (IEnumerable<float>, IEnumerable<float>) SplitArray(float[] src, uint length, uint split_index);

        void SplitArray(double[] src, double[] array1, double[] array2, uint length, uint split_index);
        (IEnumerable<double>, IEnumerable<double>) SplitArray(double[] src, uint length, uint split_index);

        /*
         * MergeArrays:
         * - int[], IEnumerable<int>
         * - long[], IEnumerable<float>
         * - float[], IEnumerable<float>
         * - double[], IEnumerable<double>
         */
        void MergeArrays(int[] result, int[] array1, int[] array2);
        IEnumerable<int> MergeArrays(int[] array1, int[] array2);

        void MergeArrays(long[] result, long[] array1, long[] array2);
        IEnumerable<long> MergeArrays(long[] array1, long[] array2);

        void MergeArrays(float[] result, float[] array1, float[] array2);
        IEnumerable<float> MergeArrays(float[] array1, float[] array2);

        void MergeArrays(double[] result, double[] array1, double[] array2);
        IEnumerable<double> MergeArrays(double[] array1, double[] array2);

        /*
         * AddArrays:
         * - int[], IEnumerable<int>
         * - long[], IEnumerable<float>
         * - float[], IEnumerable<float>
         * - double[], IEnumerable<double>
         */
        void AddArrays(int[] result, int[] array1, int[] array2);
        IEnumerable<int> AddArrays(int[] array1, int[] array2);

        void AddArrays(long[] result, long[] array1, long[] array2);
        IEnumerable<long> AddArrays(long[] array1, long[] array2);

        void AddArrays(float[] result, float[] array1, float[] array2);
        IEnumerable<float> AddArrays(float[] array1, float[] array2);

        void AddArrays(double[] result, double[] array1, double[] array2);
        IEnumerable<double> AddArrays(double[] array1, double[] array2);
    }
}
