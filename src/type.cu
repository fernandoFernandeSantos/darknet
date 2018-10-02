/*
 * type.cu
 *
 *  Created on: 01/10/2018
 *      Author: fernando
 */

extern "C" {

#include "type.h"
}
#ifdef GPU
#include "cuda.h"
#endif

/**
 * Read a file for all precisions
 */
int fread_float_to_real_t(real_t* dst, size_t siz, size_t times, FILE* fp) {
	float* temp = (float*) calloc(times, sizeof(float));
	if (temp == NULL) {
		return -1;
	}
	size_t fread_result = fread(temp, sizeof(float), times, fp);
	if (fread_result != times) {
		free(temp);
		return -1;
	}

	for (size_t i = 0; i < times; i++) {
		//TODO: make ready for half
		dst[i] = real_t(temp[i]);
	}
	free(temp);
	return fread_result;

}

#ifdef __NVCC__

__device__                __forceinline__ real_t_device exp_real(real_t_device x) {
#if REAL_TYPE == HALF
	return expf(x);
#elif REAL_TYPE == FLOAT
	return expf(x);
#elif REAL_TYPE == DOUBLE
	return exp(x);
#endif
}

__device__                __forceinline__ real_t_device floor_real(real_t_device x) {
#if REAL_TYPE == HALF
	return floorf(half(x));
#elif REAL_TYPE == FLOAT
	return floorf(x);
#elif REAL_TYPE == DOUBLE
	return floor(x);
#endif
}

__device__                __forceinline__ real_t_device pow_real(real_t_device x,
		real_t_device y) {
#if REAL_TYPE == HALF
	return powf(x, y);
#elif REAL_TYPE == FLOAT
	return powf(x, y);
#elif REAL_TYPE == DOUBLE
	return pow(x, y);
#endif
}

__device__                __forceinline__ real_t_device sqrt_real(real_t_device x) {
#if REAL_TYPE == HALF
	return sqrtf(x);
#elif REAL_TYPE == FLOAT
	return sqrtf(x);
#elif REAL_TYPE == DOUBLE
	return sqrt(x);
#endif
}

__device__                __forceinline__ real_t_device fabs_real(real_t_device x) {
#if REAL_TYPE == HALF
	return fabsf(x);
#elif REAL_TYPE == FLOAT
	return fabsf(x);
#elif REAL_TYPE == DOUBLE
	return fabs(x);
#endif
}

__device__                __forceinline__ real_t_device log_real(real_t_device x) {
#if REAL_TYPE == HALF
	return hlog(x);
#elif REAL_TYPE == FLOAT
	return logf(x);
#elif REAL_TYPE == DOUBLE
	return log(x);
#endif
}

__device__         __forceinline__ real_t_device atomic_add_real(real_t_device *x,
		real_t_device val) {
#if REAL_TYPE == HALF
#if __CUDA_ARCH__ > 700
	return atomicAdd((half*)x, (half)val);
#endif

	half old = *x;
	*x += val;
	return old;
#else
	return atomicAdd(x, val);
#endif
}

__device__        __forceinline__ real_t_device cos_real(real_t_device x) {
#if REAL_TYPE == HALF
	return hcos(x);
#elif REAL_TYPE == FLOAT
	return cosf(x);
#elif REAL_TYPE == DOUBLE
	return cos(x);
#endif
}

__device__        __forceinline__ real_t_device sin_real(real_t_device x) {
#if REAL_TYPE == HALF
	return hsin(x);
#elif REAL_TYPE == FLOAT
	return sinf(x);
#elif REAL_TYPE == DOUBLE
	return sin(x);
#endif
}

#endif
