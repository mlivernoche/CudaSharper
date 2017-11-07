#pragma once
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_profiler_api.h"
#include "device_functions.h"
#include <stdint.h>

namespace csl {

	// This class provides a simple way to figuring out the launch parameters for kernel launches.
	// In particular, the amount of blocks and threads to use. This can be defined by the subclass,
	// but generally it will be a recursive function that doubles the amount of blocks and threads
	// (the order is defined by the subclass) until they reach either max_blocks, max_threads, or
	// array_size. Kernels should not be "monolithic kernels" and should be able to iterate over an
	// arbitrary amount of data.
	// See: https://devblogs.nvidia.com/parallelforall/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
	class cuda_launch_parameters {
	protected:
		int32_t max_blocks;
		int32_t max_threads;

		virtual void determine_launch_parameters(
			int32_t* blocks, int32_t* threads,
			const int64_t array_size,
			const int32_t max_block_size, const int32_t max_thread_size) = 0;
	};

}


