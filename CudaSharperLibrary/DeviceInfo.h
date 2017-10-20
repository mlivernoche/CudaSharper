#pragma once
#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_profiler_api.h"
#include <stdint.h>
#include <atomic>
#include <cstring>

int32_t marshal_cuda_error(cudaError_t error);

class DeviceInfo {
public:
	DeviceInfo(int32_t device_id = 0);
	cudaError_t get_device_properties(int32_t device_id, cudaDeviceProp* prop) const;
	int32_t get_cuda_device_count() const;
	cudaError_t get_cuda_device_count(int32_t& result) const;
	cudaError_t get_cuda_device_name(int32_t device_id, char* device_name_ptr) const;
	cudaError_t reset_cuda_device() const;
	int32_t get_device_id() const;
private:
	static cudaDeviceProp* properties;
	static std::atomic<bool>* is_context_initialized;
	static std::atomic<bool>* is_device_prop_initialized;
	int32_t device_id;
};

extern "C" {
	__declspec(dllexport) int32_t GetCudaDeviceCount();
	__declspec(dllexport) int32_t GetCudaDeviceName(int32_t device_id, char* device_name_ptr);
	__declspec(dllexport) int32_t ResetCudaDevice();
}

