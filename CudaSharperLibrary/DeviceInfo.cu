#include "DeviceInfo.h"

int32_t marshal_cuda_error(cudaError_t error) {
	return (int)error;
}

std::atomic<bool>* DeviceInfo::is_context_initialized = new std::atomic<bool>(false);
std::atomic<bool>* DeviceInfo::is_device_prop_initialized;
cudaDeviceProp* DeviceInfo::properties;

DeviceInfo::DeviceInfo(int32_t device_id) {
	if (!DeviceInfo::is_context_initialized->load()) {
		int32_t num = this->get_cuda_device_count();
		if (num > 0) {
			DeviceInfo::is_device_prop_initialized = (std::atomic<bool>*)malloc(sizeof(std::atomic<bool>) * num);
			DeviceInfo::properties = (cudaDeviceProp*)malloc(sizeof(cudaDeviceProp) * num);

			// This loads in the CUDA context and reduces the time needed to do things like cudaMalloc (subsequent calls will be faster regardless).
			for (int i = 0; i < num; i++) {
				cudaSetDevice(i);
				cudaFree(0);
				this->get_device_properties(i, &properties[i]);
			}

			DeviceInfo::is_context_initialized->store(true);
		}
	}
	this->device_id = device_id;
}

cudaError_t DeviceInfo::get_device_properties(int32_t device_id, cudaDeviceProp *prop) const {
	/*if (DeviceInfo::is_device_prop_initialized[device_id].load() != true) {
		cudaError_t errorCode = cudaGetDeviceProperties(&DeviceInfo::properties[device_id], device_id);
		if (errorCode != cudaSuccess) return errorCode;
		DeviceInfo::is_device_prop_initialized[device_id].store(true);
	}

	*prop = DeviceInfo::properties[device_id];*/

	cudaGetDeviceProperties(prop, device_id);

	return cudaSuccess;
}

int32_t DeviceInfo::get_cuda_device_count() const {
	int32_t num = 0;
	this->get_cuda_device_count(num);
	return num;
}

cudaError_t DeviceInfo::get_cuda_device_count(int32_t& result) const {
	int num = 0;
	cudaError_t errorCode = cudaGetDeviceCount(&num);
	if (errorCode != cudaSuccess) return errorCode;

	result = num;

	return cudaSuccess;
}

cudaError_t DeviceInfo::get_cuda_device_name(int32_t device_id, char* device_name_ptr) const {
	cudaDeviceProp prop;
	// The following isn't working correctly. Eventually, it should be used, because then we can save time calling cudaGetDeviceProperties
	cudaError_t errorCode = this->get_device_properties(device_id, &prop);
	if (errorCode != cudaSuccess) return errorCode;

	// Length of cudaDeviceProp::name, according to current NVIDIA documentation: http://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html#structcudaDeviceProp_11e26f1c6bd42f4821b7ef1a4bd3bd25c
	strncpy(device_name_ptr, prop.name, 256);

	return cudaSuccess;
}

cudaError_t DeviceInfo::reset_cuda_device() const {
	cudaError_t errorCode;
	for (int i = 0; i < this->get_cuda_device_count(); i++) {
		errorCode = cudaSetDevice(i);
		if (errorCode != cudaSuccess) return errorCode;
		errorCode = cudaDeviceReset();
		if (errorCode != cudaSuccess) return errorCode;
	}
	return cudaSuccess;
}

int32_t DeviceInfo::get_device_id() const {
	return this->device_id;
}

extern "C" {
	__declspec(dllexport) int32_t GetCudaDeviceCount() {
		DeviceInfo device;
		int32_t num = 0;
		device.get_cuda_device_count(num);
		return num;
	}

	__declspec(dllexport) int32_t GetCudaDeviceName(int32_t device_id, char* device_name_ptr) {
		DeviceInfo device;
		return marshal_cuda_error(device.get_cuda_device_name(device_id, device_name_ptr));
	}

	__declspec(dllexport) int32_t ResetCudaDevice() {
		DeviceInfo device;
		return marshal_cuda_error(device.reset_cuda_device());
	}
}