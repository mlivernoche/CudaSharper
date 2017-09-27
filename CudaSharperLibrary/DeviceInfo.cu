#include "DeviceInfo.h"

extern "C" __declspec(dllexport) int GetCudaDeviceCount() {
	int num = 0;
	cudaError_t devices = cudaGetDeviceCount(&num);
	return num;
}

extern "C" __declspec(dllexport) void GetCudaDeviceName(int device_id, char* device_name_ptr) {
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, device_id);

	// Length of cudaDeviceProp::name, according to current NVIDIA documentation: http://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html#structcudaDeviceProp_11e26f1c6bd42f4821b7ef1a4bd3bd25c
	int cuda_name_length = 256;

	for (int i = 0; i < cuda_name_length; i++) {
		device_name_ptr[i] = prop.name[i];
	}
}