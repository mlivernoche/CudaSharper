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
	static cudaError_t get_cuda_device_count(int32_t& result);
	static cudaError_t intialize_cuda_context();
	static cudaError_t get_cuda_device_name(int32_t device_id, char* device_name_ptr);
	static cudaError_t reset_cuda_device();

	explicit DeviceInfo() {
		if (DeviceInfo::get_cuda_device_count() > 0) {
			DeviceInfo::is_device_prop_initialized = (std::atomic<bool>*)malloc(sizeof(std::atomic<bool>) * num);
			DeviceInfo::properties = (cudaDeviceProp*)malloc(sizeof(cudaDeviceProp) * num);
		}

		DeviceInfo::intialize_cuda_context();
	}

	static cudaError_t get_device_properties(int32_t device_id, cudaDeviceProp *prop) {
		if (DeviceInfo::is_device_prop_initialized[device_id].load() != true) {
			cudaError_t errorCode = cudaGetDeviceProperties(&DeviceInfo::properties[device_id], device_id);
			if (errorCode != cudaSuccess) return errorCode;
			DeviceInfo::is_device_prop_initialized[device_id].store(true);
		}

		*prop = DeviceInfo::properties[device_id];

		return cudaSuccess;
	}

	static cudaDeviceProp* properties;

private:
	static std::atomic<bool> is_context_initialized;
	static std::atomic<bool>* is_device_prop_initialized;
};

extern "C" {
	__declspec(dllexport) int32_t InitializeCudaContext();
	__declspec(dllexport) int32_t GetCudaDeviceCount();
	__declspec(dllexport) int32_t GetCudaDeviceName(int32_t device_id, char* device_name_ptr);
	__declspec(dllexport) int32_t ResetCudaDevice();
}

