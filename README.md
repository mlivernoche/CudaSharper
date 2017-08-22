# CudaSharper
CUDA-accelerated functions that are callable in C#.

## Introduction

CudaSharper is not intended to write CUDA in C#, but rather a library that allows one to easily use CUDA-accelerated functions without having to directly interact with the device. This library enables one to use CUDA-acceleration without knowing anything about programming in CUDA. In order to use CudaSharper, there are two components that your project will need:

CudaSharper - a wrapper for CUDA-accelerated functions. This file acts as a wrapper for CudaSharperLibrary.dll, which is required for these functions to run. CudaSharper can be a .dll that can be referenced in your C# projects, or it can be copy and pasted.

CudaSharperLibrary.dll - the actual CUDA C code compiled as a C++/CLI DLL assembly; however, it is unmanaged and therefore requires this wrapper to be used in C# projects. This must be compiled as a C++/CLI DLL assembly to be used in your project; however, because it is unmanaged, it cannot be referenced. The DLL needs to be in one of the [PATH that Windows searches in](https://en.wikipedia.org/wiki/PATH_(variable)). Calling CudaSettings.Load() will automically add AppDomain.CurrentDomain.BaseDirectory to the executable's PATH environment variables, so you can put CudaSharperLibrary.dll in the same directory for convenience.

### Example \#1: Merging two arrays

```
// Load the DLL. It only has to be called once.
CudaSettings.Load();

// We'll use the second CUDA-enabled GPU (this system has a GTX 1070 [which is 0] and a GTX 1050 Ti [which is 1]).
var cudaObject = new Cuda(1);

var array1 = Enumerable.Range(0, 10_000);
var array2 = Enumerable.Range(0, 10_000);

// This takes arrays, but it returns IEnumerable<T> (see below for supported types).
var merged_array = cudaObject.MergeArrays(array1.ToArray(), array2.ToArray());
```

### Example \#2: Generating random numbers

```
// Load the DLL. It only has to be called once.
CudaSettings.Load();

// We'll use the first CUDA-enabled GPU (this system has a GTX 1070 [which is 0] and a GTX 1050 Ti [which is 1]).
var cudaObject = new CuRand(0);

// Generate 100,000 random numbers using a uniform distribution. The return value is IEnumerable<float>.
var uniform_rand = cuRand.GenerateUniformDistribution(100_000);
```

### When to use CPU vs GPU
The CUDA programming model allows easy scaling of performance. However, due to the high latency of the global memory (e.g., GDDR5), the GPU is designed to have dozens of active threads per SM at any time to combat the high latency. The GPU has to be swarmed with threads to ensure the cores are being feed at all times. In other words, smaller work loads (e.g., generating 20 random numbers) will be faster on the CPU than on the GPU. The GPU performs best in large work loads (e.g. generating 50,000 random numbers).

### CUDA version

This was developed with CUDA toolkit 8.0 and two Pascal GPUs, GTX 1070 and GTX 1050 Ti. Support for older GPUs has not been tested, but they are possible.

## CudaSettings

CudaSettings.Load(): This function is meant to set the directory of the executeable. This is for loading CudaSharperLibrary.dll in CudaSharper.

## Current Functions

### Construction of CudaSharper classes

You have to specify the ID of the CUDA-enabled device to use when creating an object. The ID is simply an int, and starts at 0. You can also call GetCudaDeviceName(int device_id) to get the name of the CUDA device.

### Cuda
1. SplitArray: Takes one array, and returns a tuple of each half. Supports int, float, long, double.
2. AddArrays: Takes two arrays, adds them together, and returns the result. Supports int, float, long, double.
3. MergeArrays: Takes two arrays, and returns a combined array. Supports int, float, long, double.

### CuRand
Allows generating random numbers with the cuRAND library. These should be used for situations that require a large amount of random numbers; for example, on a GTX 1050 Ti, curand_uniform can generate 50,000 random numbers in about 10-13 milliseconds.

| cuRAND Distribution | CudaSharper Method |
| ------------------- | ------------------ |
| curand_uniform      | GenerateUniformDistribution |
| curand_normal       | GenerateNormalDistribution |
| curand_log_normal   | GenerateLogNormalDistribution |
| curand_poisson      | GeneratePoissonDistribution (CUDA written, wrapper not) |
| curand_uniform_double | TODO |
| curand_normal_double | TODO |
| curand_log_normal_double | TODO |

The functions that generate double (e.g., curand_normal2) and quadruple (e.g., curand_normal4) tuples will be implemented seperately.

## Performance

Performance is very good, as far as I can tell. These kernels were written and tested on a GTX 1050 Ti. Performance was profiled by NVIDIA Visual Profiler. CUDA is best used for large amounts of small data (e.g., tens of thousands or millions of ints) and simple computational operations. For example, if you just need to generate 10 random ints, then just use System.Random.

These kernels are written in such a way that bigger GPUs should be able to make use of more blocks and threads. So, running these kernels on a 1080 Ti should be much faster than on the GTX 1050 Ti I am using.

## Changelog
08/18/2017 - greatly improved the performance of the cuRAND functions.

08/17/2017 - added support for float, long, and double to SplitArray; added MergeArrays.
