#pragma once
#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_profiler_api.h"

extern "C" __declspec(dllexport) int GetCudaDeviceCount();
extern "C" __declspec(dllexport) void GetCudaDeviceName(int device_id, char* device_name_ptr);

