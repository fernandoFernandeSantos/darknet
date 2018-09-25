/*
 * type.h
 *
 *  Created on: 13/09/2018
 *      Author: fernando
 */

#ifndef TYPE_H_
#define TYPE_H_

#define HALF 16
#define FLOAT 32
#define DOUBLE 64

#include <stdlib.h>
#include <stdio.h>

#if REAL_TYPE == HALF

// For half precision
#include <cuda_fp16.h>
#include <cuda_fp16.hpp>
//#include "half.hpp"

//HALF-----------------------------------------------------------------------------------------------
//typedef half_float::half real_t;
typedef half real_t;

typedef half real_t_device;

#define FLT_MAX real_t(65504 - 1)

#define CAST(a) real_t_device(float(a))

#define REAL_INFINITY 0x7C00

void transform_float_to_half_array(real_t_device* dst, float* src, size_t n);

//---------------------------------------------------------------------------------------------------

#elif REAL_TYPE == FLOAT

//FLOAT----------------------------------------------------------------------------------------------
// Single precision
typedef float real_t;
typedef real_t real_t_device;

#define FLT_MAX real_t(1E+37)

#define CAST(a) (a)

#define REAL_INFINITY 0x7F800000
//---------------------------------------------------------------------------------------------------

#elif REAL_TYPE == DOUBLE

//DOUBLE----------------------------------------------------------------------------------------------
//Double precision
typedef double real_t;
typedef real_t real_t_device;

#define FLT_MAX real_t(1E+307)
#define CAST(a) (a)


#define REAL_INFINITY 0x7FF0000000000000

//---------------------------------------------------------------------------------------------------
#endif

#ifdef __NVCC__

typedef struct __device_builtin__ {
	real_t_device x;
	real_t_device y;
	real_t_device z;
}real_t3;

#endif

__device__         __forceinline__ real_t_device exp_real(real_t_device x) {
#if REAL_TYPE == HALF
	return hexp(x);
#elif REAL_TYPE == FLOAT
	return expf(x);
#elif REAL_TYPE == DOUBLE
	return exp(x);
#endif
}

__device__         __forceinline__ real_t_device floor_real(real_t_device x) {
#if REAL_TYPE == HALF
	return hfloor(half(x));
#elif REAL_TYPE == FLOAT
	return floorf(x);
#elif REAL_TYPE == DOUBLE
	return floor(x);
#endif
}

__device__         __forceinline__ real_t_device pow_real(real_t_device x,
		real_t_device y) {
#if REAL_TYPE == HALF
	return real_t_device(powf(float(x), y));
#elif REAL_TYPE == FLOAT
	return powf(x, y);
#elif REAL_TYPE == DOUBLE
	return pow(x, y);
#endif
}

__device__         __forceinline__ real_t_device sqrt_real(real_t_device x) {
#if REAL_TYPE == HALF
	return hsqrt(x);
#elif REAL_TYPE == FLOAT
	return sqrtf(x);
#elif REAL_TYPE == DOUBLE
	return sqrt(x);
#endif
}

__device__         __forceinline__ real_t_device fabs_real(real_t_device x) {
#if REAL_TYPE == HALF
	return fabsf(x);
#elif REAL_TYPE == FLOAT
	return fabsf(x);
#elif REAL_TYPE == DOUBLE
	return fabs(x);
#endif
}

__device__         __forceinline__ real_t_device log_real(real_t_device x) {
#if REAL_TYPE == HALF
	return hlog(x);
#elif REAL_TYPE == FLOAT
	return logf(x);
#elif REAL_TYPE == DOUBLE
	return log(x);
#endif
}

__device__  __forceinline__ real_t_device atomic_add_real(real_t_device *x,
		real_t_device val) {
//
//	uint16* address_as_ull = (uint16*)x;
//	uint16 old = *address_as_ull, assumed;
//
//	do {
//		assumed = old;
//		old = atomicCAS(address_as_ull, assumed,
//				__half_as_short(val +
//						__short_as_half(assumed)));
//
//		// Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
//	}while (assumed != old);
//
//	return __longlong_as_double(old);
//#else
#if __CUDA_ARCH__ > 700
	return atomicAdd(x, val);
#else
	return 0;
#endif
}

__device__ __forceinline__ real_t_device cos_real(real_t_device x){
#if REAL_TYPE == HALF
	return hcos(x);
#elif REAL_TYPE == FLOAT
	return cosf(x);
#elif REAL_TYPE == DOUBLE
	return cos(x);
#endif
}

__device__ __forceinline__ real_t_device sin_real(real_t_device x){
#if REAL_TYPE == HALF
	return hsin(x);
#elif REAL_TYPE == FLOAT
	return sinf(x);
#elif REAL_TYPE == DOUBLE
	return sin(x);
#endif
}

int fread_float_to_real_t(real_t* dst, size_t siz, size_t times, FILE* fp);

#endif /* TYPE_H_ */
