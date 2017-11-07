#pragma once
#include "device_info.h"

namespace csl {

	int32_t marshal_cuda_error(cudaError_t error) {
		return (int)error;
	}

	device_info::device_info(int32_t device_id) {
		this->device_id = device_id;
	}

	cudaError_t device_info::get_device_properties(int32_t device_id, cudaDeviceProp *prop) const {
		cudaGetDeviceProperties(prop, device_id);

		return cudaSuccess;
	}

	int32_t device_info::get_cuda_device_count() const {
		int32_t num = 0;
		this->get_cuda_device_count(num);
		return num;
	}

	cudaError_t device_info::get_cuda_device_count(int32_t& result) const {
		int num = 0;
		cudaError_t errorCode = cudaGetDeviceCount(&num);
		if (errorCode != cudaSuccess) return errorCode;

		result = num;

		return cudaSuccess;
	}

	cudaError_t device_info::get_cuda_device_name(int32_t device_id, char* device_name_ptr) const {
		cudaDeviceProp prop;
		// The following isn't working correctly. Eventually, it should be used, because then we can save time calling cudaGetDeviceProperties
		cudaError_t errorCode = this->get_device_properties(device_id, &prop);
		if (errorCode != cudaSuccess) return errorCode;

		// Length of cudaDeviceProp::name, according to current NVIDIA documentation: http://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html#structcudaDeviceProp_11e26f1c6bd42f4821b7ef1a4bd3bd25c
		strncpy(device_name_ptr, prop.name, 256);

		return cudaSuccess;
	}

	cudaError_t device_info::reset_cuda_device() const {
		cudaError_t errorCode;
		for (int i = 0; i < this->get_cuda_device_count(); i++) {
			errorCode = cudaSetDevice(i);
			if (errorCode != cudaSuccess) return errorCode;
			errorCode = cudaDeviceReset();
			if (errorCode != cudaSuccess) return errorCode;
		}
		return cudaSuccess;
	}

	int32_t device_info::get_device_id() const {
		return this->device_id;
	}

}

