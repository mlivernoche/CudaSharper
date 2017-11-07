#pragma once
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_profiler_api.h"
#include "device_functions.h"
#include "thrust\device_vector.h"
#include <stdint.h>
#include <memory>

namespace csl {
	// This is a wrapper for allocating and freeing memory.
	// When you reference an instance of this, it tries to use the same allocated space of memory,
	// rather than allocating and freeing it every single time. This does NOT support reallocating
	// memory. This class is also "lazy": it does not allocate the memory until it is needed.
	template<typename T>
	class cuda_device_ptr {
	public:

		cuda_device_ptr(int32_t device_id, int64_t size_of_array) {
			this->cuda_device_id = device_id;
			this->heap_size = size_of_array;
		}

		~cuda_device_ptr() {
			if (this->ptr != NULL) {
				cudaSetDevice(this->cuda_device_id);
				this->ptr.reset();
			}
		}

		__inline__ void force_allocate() {
			if (this->ptr == NULL) {
				cudaSetDevice(this->cuda_device_id);
				this->ptr = std::make_shared<thrust::device_vector<T>>(this->heap_size);
			}
		}

		// Gets the pointer of the device memory. This function is thread-safe.
		__inline__ T* get() {
			this->force_allocate();

			return thrust::raw_pointer_cast(this->ptr->data());
		}

		__inline__ int64_t max_size() const {
			return this->heap_size;
		}
	private:
		std::shared_ptr<thrust::device_vector<T>> ptr;
		int32_t cuda_device_id;
		int64_t heap_size;
	};

	// This class allows one to use four instances of cuda_alloc_ptr: int32_t, int64_t, float, and double.
	// Adding types to this class should be CLS-compliant for use in .NET (e.g., no float4).
	// These should include types commonly used in CUDA and are CLS-compliant.
	// This class uses lazy initialization, in that it won't allocate memory for each type unless a pointer for
	// it is requested. In other words, adding types to this class is fine, because any types that aren't used
	// won't be allocated in device memory. Use force_allocate() if you want to allocate memory for each type at once,
	// regardless if it will be used later.
	class cuda_device {
	public:
		cuda_device(int32_t device_id, int32_t alloc_size) {
			this->cuda_device_id = device_id;
			this->heap_size = alloc_size;
		}

		~cuda_device() {
			this->device_ptr_u32.reset();
			this->device_ptr_u64.reset();
			this->device_ptr_f32.reset();
			this->device_ptr_f64.reset();
		}

		// cuda_device_ptr does not allocate memory until a pointer to it is requested,
		// which allows the creation of a lot of them quickly and then allocating memory as they are created.
		// If you need to explcitly allocate memory, call this function.
		__inline__ void force_allocate() {
			if (this->device_ptr_u32 == NULL) {
				device_ptr_u32 = std::make_shared<cuda_device_ptr<int32_t>>(this->cuda_device_id, this->heap_size);
				device_ptr_u32->force_allocate();
			}

			if (this->device_ptr_u64 == NULL) {
				device_ptr_u64 = std::make_shared<cuda_device_ptr<int64_t>>(this->cuda_device_id, this->heap_size);
				device_ptr_u64->force_allocate();
			}

			if (this->device_ptr_f32 == NULL) {
				device_ptr_f32 = std::make_shared<cuda_device_ptr<float>>(this->cuda_device_id, this->heap_size);
				device_ptr_f32->force_allocate();
			}

			if (this->device_ptr_f64 == NULL) {
				device_ptr_f64 = std::make_shared<cuda_device_ptr<double>>(this->cuda_device_id, this->heap_size);
				device_ptr_f64->force_allocate();
			}
		}

		__inline__ int32_t* getu32() {
			if (this->device_ptr_u32 == NULL) {
				device_ptr_u32 = std::make_shared<cuda_device_ptr<int32_t>>(this->cuda_device_id, this->heap_size);
			}

			return this->device_ptr_u32->get();
		}

		__inline__ int64_t* getu64() {
			if (this->device_ptr_u64 == NULL) {
				device_ptr_u64 = std::make_shared<cuda_device_ptr<int64_t>>(this->cuda_device_id, this->heap_size);
			}

			return this->device_ptr_u64->get();
		}

		__inline__ float* getf32() {
			if (this->device_ptr_f32 == NULL) {
				device_ptr_f32 = std::make_shared<cuda_device_ptr<float>>(this->cuda_device_id, this->heap_size);
			}

			return this->device_ptr_f32->get();
		}

		__inline__ double* getf64() {
			if (this->device_ptr_f64 == NULL) {
				device_ptr_f64 = std::make_shared<cuda_device_ptr<double>>(this->cuda_device_id, this->heap_size);
			}

			return this->device_ptr_f64->get();
		}

		__inline__ int32_t max_size() const {
			return this->heap_size;
		}

		__inline__ int32_t device_id() const {
			return this->cuda_device_id;
		}

	private:
		int32_t cuda_device_id;
		int32_t heap_size;
		std::shared_ptr<cuda_device_ptr<int32_t>> device_ptr_u32;
		std::shared_ptr<cuda_device_ptr<int64_t>> device_ptr_u64;
		std::shared_ptr<cuda_device_ptr<float>> device_ptr_f32;
		std::shared_ptr<cuda_device_ptr<double>> device_ptr_f64;
	};
}
