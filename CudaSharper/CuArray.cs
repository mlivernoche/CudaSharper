using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

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
    public class CuArray : ICudaDevice
    {
        private ICudaDevice CudaDeviceComponent { get; }

        public int DeviceId => CudaDeviceComponent.DeviceId;

        static CuArray()
        {
            CudaSettings.Load();
        }

        public CuArray(int device_id)
        {
            CudaDeviceComponent = new CudaDevice(device_id);
        }

        public string GetCudaDeviceName()
        {
            return CudaDeviceComponent.GetCudaDeviceName();
        }

        [DllImport("CudaSharperLibrary.dll")]
        private static extern void SplitIntArray(int device_id, int[] src, int[] array1, int[] array2, uint length, uint split_index);

        public void Split(int[] src, int[] array1, int[] array2, uint length, uint split_index)
        {
            if (split_index >= length)
                throw new ArgumentOutOfRangeException(string.Format("Split index is greater then the length of the array. Supplied values: length={0}, split_index={1}.", length, split_index));

            SplitIntArray(DeviceId, src, array1, array2, length, split_index);
        }

        public (int[] Left, int[] Right) Split(int[] src, uint length, uint split_index)
        {
            if (split_index >= length)
                throw new ArgumentOutOfRangeException(string.Format("Split index is greater then the length of the array. Supplied values: length={0}, split_index={1}.", length, split_index));

            var array1 = new int[split_index];
            var array2 = new int[length - split_index];
            SplitIntArray(DeviceId, src, array1, array2, length, split_index);
            return (array1, array2);
        }

        [DllImport("CudaSharperLibrary.dll")]
        private static extern void SplitLongArray(int device_id, long[] src, long[] array1, long[] array2, uint length, uint split_index);

        public void Split(long[] src, long[] array1, long[] array2, uint length, uint split_index)
        {
            if (split_index >= length)
                throw new ArgumentOutOfRangeException(string.Format("Split index is greater then the length of the array. Supplied values: length={0}, split_index={1}.", length, split_index));

            SplitLongArray(DeviceId, src, array1, array2, length, split_index);
        }

        public (long[] Left, long[] Right) Split(long[] src, uint length, uint split_index)
        {
            if (split_index >= length)
                throw new ArgumentOutOfRangeException(string.Format("Split index is greater then the length of the array. Supplied values: length={0}, split_index={1}.", length, split_index));

            var array1 = new long[split_index];
            var array2 = new long[length - split_index];
            SplitLongArray(DeviceId, src, array1, array2, length, split_index);
            return (array1, array2);
        }

        [DllImport("CudaSharperLibrary.dll")]
        private static extern void SplitFloatArray(int device_id, float[] src, float[] array1, float[] array2, uint length, uint split_index);

        public void Split(float[] src, float[] array1, float[] array2, uint length, uint split_index)
        {
            if (split_index >= length)
                throw new ArgumentOutOfRangeException(string.Format("Split index is greater then the length of the array. Supplied values: length={0}, split_index={1}.", length, split_index));

            SplitFloatArray(DeviceId, src, array1, array2, length, split_index);
        }

        public (float[] Left, float[] Right) Split(float[] src, uint length, uint split_index)
        {
            if (split_index >= length)
                throw new ArgumentOutOfRangeException(string.Format("Split index is greater then the length of the array. Supplied values: length={0}, split_index={1}.", length, split_index));

            var array1 = new float[split_index];
            var array2 = new float[length - split_index];
            SplitFloatArray(DeviceId, src, array1, array2, length, split_index);
            return (array1, array2);
        }

        [DllImport("CudaSharperLibrary.dll")]
        private static extern void SplitDoubleArray(int device_id, double[] src, double[] array1, double[] array2, uint length, uint split_index);

        public void Split(double[] src, double[] array1, double[] array2, uint length, uint split_index)
        {
            if (split_index >= length)
                throw new ArgumentOutOfRangeException(string.Format("Split index is greater then the length of the array. Supplied values: length={0}, split_index={1}.", length, split_index));

            SplitDoubleArray(DeviceId, src, array1, array2, length, split_index);
        }

        public (double[] Left, double[] Right) Split(double[] src, uint length, uint split_index)
        {
            if (split_index >= length)
                throw new ArgumentOutOfRangeException(string.Format("Split index is greater then the length of the array. Supplied values: length={0}, split_index={1}.", length, split_index));

            var array1 = new double[split_index];
            var array2 = new double[length - split_index];
            SplitDoubleArray(DeviceId, src, array1, array2, length, split_index);
            return (array1, array2);
        }

        [DllImport("CudaSharperLibrary.dll")]
        private static extern void MergeIntArrays(int device_id, int[] result, int[] array1, int[] array2, uint array1_length, uint array2_length);

        public void Merge(int[] result, int[] array1, int[] array2)
        {
            if (array1.Length + array2.Length > Int32.MaxValue)
                throw new Exception("Resultant array is too large.");

            MergeIntArrays(DeviceId, result, array1, array2, (uint)array1.Length, (uint)array2.Length);
        }

        public int[] Merge(int[] array1, int[] array2)
        {
            if (array1.Length + array2.Length > Int32.MaxValue)
                throw new Exception("Resultant array is too large.");

            var result = new int[array1.Length + array2.Length];
            MergeIntArrays(DeviceId, result, array1, array2, (uint)array1.Length, (uint)array2.Length);
            return result;
        }

        [DllImport("CudaSharperLibrary.dll")]
        private static extern void MergeLongArrays(int device_id, long[] result, long[] array1, long[] array2, uint array1_length, uint array2_length);

        public void Merge(long[] result, long[] array1, long[] array2)
        {
            if (array1.Length + array2.Length > Int32.MaxValue)
                throw new Exception("Resultant array is too large.");

            MergeLongArrays(DeviceId, result, array1, array2, (uint) array1.Length, (uint) array2.Length);
        }

        public long[] Merge(long[] array1, long[] array2)
        {
            if (array1.Length + array2.Length > Int32.MaxValue)
                throw new Exception("Resultant array is too large.");

            var result = new long[array1.Length + array2.Length];
            MergeLongArrays(DeviceId, result, array1, array2, (uint) array1.Length, (uint) array2.Length);
            return result;
        }

        [DllImport("CudaSharperLibrary.dll")]
        private static extern void MergeFloatArrays(int device_id, float[] result, float[] array1, float[] array2, int array1_length, int array2_length);

        public void Merge(float[] result, float[] array1, float[] array2)
        {
            if (array1.Length + array2.Length > Int32.MaxValue)
                throw new Exception("Resultant array is too large.");

            MergeFloatArrays(DeviceId, result, array1, array2, array1.Length, array2.Length);
        }

        public float[] Merge(float[] array1, float[] array2)
        {
            if (array1.Length + array2.Length > Int32.MaxValue)
                throw new Exception("Resultant array is too large.");

            var result = new float[array1.Length + array2.Length];
            MergeFloatArrays(DeviceId, result, array1, array2, array1.Length, array2.Length);
            return result;
        }

        [DllImport("CudaSharperLibrary.dll")]
        private static extern void MergeDoubleArrays(int device_id, double[] result, double[] array1, double[] array2, int array1_length, int array2_length);

        public void Merge(double[] result, double[] array1, double[] array2)
        {
            if (array1.Length + array2.Length > Int32.MaxValue)
                throw new Exception("Resultant array is too large.");

            MergeDoubleArrays(DeviceId, result, array1, array2, array1.Length, array2.Length);
        }

        public double[] Merge(double[] array1, double[] array2)
        {
            if (array1.Length + array2.Length > Int32.MaxValue)
                throw new Exception("Resultant array is too large.");

            var result = new double[array1.Length + array2.Length];
            MergeDoubleArrays(DeviceId, result, array1, array2, array1.Length, array2.Length);
            return result;
        }

        [DllImport("CudaSharperLibrary.dll")]
        private static extern void AddIntArrays(int device_id, int[] result, int[] array1, int[] array2, int length);

        public void Add(int[] result, int[] array1, int[] array2)
        {
            if (array1.Length != result.Length || array2.Length != result.Length)
                throw new ArgumentOutOfRangeException("Bad arrays given; they need to be the same length.");

            AddIntArrays(DeviceId, result, array1, array2, array1.Length);
        }

        public int[] Add(int[] array1, int[] array2)
        {
            if (array1.Length != array2.Length)
                throw new ArgumentOutOfRangeException("Bad arrays given; they need to be the same length.");

            var result = new int[array1.Length];
            AddIntArrays(DeviceId, result, array1, array2, array1.Length);
            return result;
        }

        [DllImport("CudaSharperLibrary.dll")]
        private static extern void AddFloatArrays(int device_id, float[] result, float[] array1, float[] array2, int length);

        public void Add(float[] result, float[] array1, float[] array2)
        {
            if (array1.Length != result.Length || array2.Length != result.Length)
                throw new ArgumentOutOfRangeException("Bad arrays given; they need to be the same length.");

            AddFloatArrays(DeviceId, result, array1, array2, array1.Length);
        }

        public float[] Add(float[] array1, float[] array2)
        {
            if (array1.Length != array2.Length)
                throw new ArgumentOutOfRangeException("Bad arrays given; they need to be the same length.");

            var result = new float[array1.Length];
            AddFloatArrays(DeviceId, result, array1, array2, array1.Length);
            return result;
        }

        [DllImport("CudaSharperLibrary.dll")]
        private static extern void AddLongArrays(int device_id, long[] result, long[] array1, long[] array2, int length);

        public void Add(long[] result, long[] array1, long[] array2)
        {
            if (array1.Length != result.Length || array2.Length != result.Length)
                throw new ArgumentOutOfRangeException("Bad arrays given; they need to be the same length.");

            AddLongArrays(DeviceId, result, array1, array2, array1.Length);
        }

        public long[] Add(long[] array1, long[] array2)
        {
            if (array1.Length != array2.Length)
                throw new Exception("Bad arrays given; they need to be the same length.");

            var result = new long[array1.Length];
            AddLongArrays(DeviceId, result, array1, array2, array1.Length);
            return result;
        }

        [DllImport("CudaSharperLibrary.dll")]
        private static extern void AddDoubleArrays(int device_id, double[] result, double[] array1, double[] array2, int length);

        public void Add(double[] result, double[] array1, double[] array2)
        {
            if (array1.Length != result.Length || array2.Length != result.Length)
                throw new Exception("Bad arrays given; they need to be the same length.");

            AddDoubleArrays(DeviceId, result, array1, array2, array1.Length);
        }

        public double[] Add(double[] array1, double[] array2)
        {
            if (array1.Length != array2.Length)
                throw new Exception("Bad arrays given; they need to be the same length.");

            var result = new double[array1.Length];
            AddDoubleArrays(DeviceId, result, array1, array2, array1.Length);
            return result;
        }

        [DllImport("CudaSharperLibrary.dll")]
        private static extern void SubtractIntArrays(int device_id, int[] result, int[] array1, int[] array2, int length);

        public void Subtract(int[] result, int[] array1, int[] array2)
        {
            if (array1.Length != result.Length || array2.Length != result.Length)
                throw new ArgumentOutOfRangeException("Bad arrays given; they need to be the same length.");

            SubtractIntArrays(DeviceId, result, array1, array2, array1.Length);
        }

        public int[] Subtract(int[] array1, int[] array2)
        {
            if (array1.Length != array2.Length)
                throw new ArgumentOutOfRangeException("Bad arrays given; they need to be the same length.");

            var result = new int[array1.Length];
            SubtractIntArrays(DeviceId, result, array1, array2, array1.Length);
            return result;
        }

        [DllImport("CudaSharperLibrary.dll")]
        private static extern void SubtractFloatArrays(int device_id, float[] result, float[] array1, float[] array2, int length);

        public void Subtract(float[] result, float[] array1, float[] array2)
        {
            if (array1.Length != result.Length || array2.Length != result.Length)
                throw new ArgumentOutOfRangeException("Bad arrays given; they need to be the same length.");

            SubtractFloatArrays(DeviceId, result, array1, array2, array1.Length);
        }

        public float[] Subtract(float[] array1, float[] array2)
        {
            if (array1.Length != array2.Length)
                throw new ArgumentOutOfRangeException("Bad arrays given; they need to be the same length.");

            var result = new float[array1.Length];
            SubtractFloatArrays(DeviceId, result, array1, array2, array1.Length);
            return result;
        }

        [DllImport("CudaSharperLibrary.dll")]
        private static extern void SubtractLongArrays(int device_id, long[] result, long[] array1, long[] array2, int length);

        public void Subtract(long[] result, long[] array1, long[] array2)
        {
            if (array1.Length != result.Length || array2.Length != result.Length)
                throw new ArgumentOutOfRangeException("Bad arrays given; they need to be the same length.");

            SubtractLongArrays(DeviceId, result, array1, array2, array1.Length);
        }

        public long[] Subtract(long[] array1, long[] array2)
        {
            if (array1.Length != array2.Length)
                throw new Exception("Bad arrays given; they need to be the same length.");

            var result = new long[array1.Length];
            SubtractLongArrays(DeviceId, result, array1, array2, array1.Length);
            return result;
        }

        [DllImport("CudaSharperLibrary.dll")]
        private static extern void SubtractDoubleArrays(int device_id, double[] result, double[] array1, double[] array2, int length);

        public void Subtract(double[] result, double[] array1, double[] array2)
        {
            if (array1.Length != result.Length || array2.Length != result.Length)
                throw new Exception("Bad arrays given; they need to be the same length.");

            SubtractDoubleArrays(DeviceId, result, array1, array2, array1.Length);
        }

        public double[] Subtract(double[] array1, double[] array2)
        {
            if (array1.Length != array2.Length)
                throw new Exception("Bad arrays given; they need to be the same length.");

            var result = new double[array1.Length];
            SubtractDoubleArrays(DeviceId, result, array1, array2, array1.Length);
            return result;
        }
    }
}
