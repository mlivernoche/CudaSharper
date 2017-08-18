# CudaSharper
CUDA-accelerated functions that are callable in C#.

## Introduction

CudaSharper is not intended to write CUDA in C#, but rather a library that allows one to easily use CUDA-accelerated functions without having to directly interact with the device. This library enables one to use CUDA-acceleration without knowing anything about programming in CUDA. In order to use CudaSharper, there are two components that your project will need:

CudaSharper - a wrapper for CUDA-accelerated functions. This file acts as a wrapper for CudaSharperLibrary.dll, which is required for these functions to run. CudaSharper can be a .dll that can be referenced in your C# projects, or it can be copy and pasted.

CudaSharperLibrary.dll - the actual CUDA C code compiled as a C++/CLI assembly; however, it is unmanaged and therefore requires this wrapper to be used in C# projects. This must be compiled as a C++/CLI assembly to be used in your project; however, because it is unmanaged, it cannot be referenced.

## CudaSettings

If you have an extra GPU (which I recommend when one is developing with CUDA), you can specify which CUDA-enabled device to use.

CudaSettings.DeviceId: The ID of the device. 0 is the default CUDA-enabled device, 1 is the next one, etc. For example, in my system, my main GPU is a GTX 1070 (which is 0) and the second GPU is a GTX 1050 Ti (which is 1).

CudaSettings.Load(): This function is meant to set the directory of the executeable. This is for loading CudaSharperLibrary.dll in CudaSharper.

## Current Functions

### Cuda
1. SplitArray: Takes one array, and returns a tuple of each half. Supports int, float, long, double.
2. AddArrays: Takes two arrays, adds them together, and returns the result. Supports int, float, long, double.
3. MergeArrays: Takes two arrays, and returns a combined array. Supports int, float, long, double.

### CuRand
1. GenerateUniformDistribution: Generates a collection of floats generated by curand_uniform.
2. GenerateNormalDistribution: Generates a collection of floats generated by curand_normal.
3. GeneratePoissonDistribution: Generates a collection of ints generated by curand_poisson.

## Performance

Performance is very good, as far as I can tell. These kernels were written and tested on a GTX 1050 Ti. Performance was profiled by NVIDIA Visual Profiler. CUDA is best used for large amounts of small data (e.g., tens of thousands or millions of ints) and simple computational operations. For example, if you just need to generate 10 random ints, then just use System.Random.

These kernels are written in such a way that bigger GPUs should be able to make use of more blocks and threads. So, running these kernels on a 1080 Ti should be much faster than on the GTX 1050 Ti I am using.

Changelog
08/18/2017 - greatly improved the performance of the cuRAND functions.
