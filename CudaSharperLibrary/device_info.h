#pragma once
#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_profiler_api.h"
#include "thrust\host_vector.h"
#include <stdint.h>
#include <atomic>
#include <cstring>
#include <memory>
#include <vector>

namespace csl {

	int32_t marshal_cuda_error(cudaError_t error);

	class device_info {
	public:
		device_info(int32_t device_id = 0);
		cudaError_t get_device_properties(int32_t device_id, cudaDeviceProp* prop) const;
		int32_t get_cuda_device_count() const;
		cudaError_t get_cuda_device_count(int32_t& result) const;
		cudaError_t get_cuda_device_name(int32_t device_id, char* device_name_ptr) const;
		cudaError_t reset_cuda_device() const;
		int32_t get_device_id() const;
	private:
		int32_t device_id;
	};

}

