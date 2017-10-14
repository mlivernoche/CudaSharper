#include "DeviceInfo.h"

int32_t marshal_cuda_error(cudaError_t error) {
	return (int)error;
}

std::atomic<bool> DeviceInfo::is_context_initialized(false);
std::atomic<bool>* DeviceInfo::is_device_prop_initialized;
cudaDeviceProp* DeviceInfo::properties;

cudaError_t DeviceInfo::get_cuda_device_count(int32_t& result) {
	int num = 0;
	cudaError_t errorCode = cudaGetDeviceCount(&num);
	if (errorCode != cudaSuccess) return errorCode;

	result = num;

	return cudaSuccess;
}

cudaError_t DeviceInfo::intialize_cuda_context() {

	if (DeviceInfo::is_context_initialized.load() != true) {
		int num = 0;
		DeviceInfo::get_cuda_device_count(num);

		cudaError_t errorCode;

		if (num > 0) {
			DeviceInfo::is_device_prop_initialized = (std::atomic<bool>*)malloc(sizeof(std::atomic<bool>) * num);
			DeviceInfo::properties = (cudaDeviceProp*)malloc(sizeof(cudaDeviceProp) * num);

			// This loads in the CUDA context and reduces the time needed to do things like cudaMalloc (subsequent calls will be faster regardless).
			for (int i = 0; i < GetCudaDeviceCount(); i++) {
				errorCode = cudaSetDevice(i);
				if (errorCode != cudaSuccess) return errorCode;
				errorCode = cudaFree(0);
				if (errorCode != cudaSuccess) return errorCode;
			}
		}
		DeviceInfo::is_context_initialized.store(true);
	}

	return cudaSuccess;
}

cudaError_t DeviceInfo::get_cuda_device_name(int32_t device_id, char* device_name_ptr) {
	cudaDeviceProp prop;
	cudaError_t errorCode = DeviceInfo::get_device_properties(device_id, &prop);
	if (errorCode != cudaSuccess) return errorCode;

	// Length of cudaDeviceProp::name, according to current NVIDIA documentation: http://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html#structcudaDeviceProp_11e26f1c6bd42f4821b7ef1a4bd3bd25c
	strncpy(device_name_ptr, prop.name, 256);

	return cudaSuccess;
}

cudaError_t DeviceInfo::reset_cuda_device() {
	cudaError_t errorCode;
	for (int i = 0; i < GetCudaDeviceCount(); i++) {
		errorCode = cudaSetDevice(i);
		if (errorCode != cudaSuccess) return errorCode;
		errorCode = cudaDeviceReset();
		if (errorCode != cudaSuccess) return errorCode;
	}
	return cudaSuccess;
}

extern "C" {
	__declspec(dllexport) int32_t InitializeCudaContext() {
		return marshal_cuda_error(DeviceInfo::intialize_cuda_context());
	}

	__declspec(dllexport) int32_t GetCudaDeviceCount() {
		int num = 0;
		DeviceInfo::get_cuda_device_count(num);
		return num;
	}

	__declspec(dllexport) int32_t GetCudaDeviceName(int32_t device_id, char* device_name_ptr) {
		return marshal_cuda_error(DeviceInfo::get_cuda_device_name(device_id, device_name_ptr));
	}

	__declspec(dllexport) int32_t ResetCudaDevice() {
		return marshal_cuda_error(DeviceInfo::reset_cuda_device());
	}
}