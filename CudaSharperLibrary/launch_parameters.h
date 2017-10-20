#pragma once
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_profiler_api.h"
#include "device_functions.h"
#include <stdint.h>

class cuda_launch_parameters {
protected:
	int32_t max_blocks;
	int32_t max_threads;

	virtual void determine_launch_parameters(
		int32_t* blocks, int32_t* threads,
		const int64_t array_size,
		const int32_t max_block_size, const int32_t max_thread_size) = 0;
};
