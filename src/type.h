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
#include "half.hpp"

typedef half_float::half real_t;

typedef struct real_t_device : __half {
// real_t_device;

	__host__ __forceinline__ friend half operator*(const __half &lh, const int &rh) {
		float t = float(lh) * rh;
		return half(t);
	}
};

#define FLT_MAX real_t(65504 - 1)

void transform_float_to_half_array(real_t_device* dst, float* src, size_t n);

#elif REAL_TYPE == FLOAT
// Single precision
typedef float real_t;
typedef real_t real_t_device;

#define FLT_MAX real_t(1E+37)

#elif REAL_TYPE == DOUBLE
//Double precision
typedef double real_t;
typedef real_t real_t_device;

#define FLT_MAX real_t(1E+307)

#endif

#ifdef __NVCC__

typedef struct __device_builtin__ {
	real_t x;
	real_t y;
	real_t z;
}real_t3;

#endif

__device__   inline real_t_device exp_real(real_t_device x) {
#if REAL_TYPE == HALF
	return hexp(x);
#elif REAL_TYPE == FLOAT
	return expf(x);
#elif REAL_TYPE == DOUBLE
	return exp(x);
#endif
}

__device__   inline real_t_device floor_real(real_t_device x) {
#if REAL_TYPE == HALF
	return hfloor(x);
#elif REAL_TYPE == FLOAT
	return floorf(x);
#elif REAL_TYPE == DOUBLE
	return floor(x);
#endif
}

__device__   inline real_t_device pow_real(real_t_device x, long y) {
#if REAL_TYPE == HALF
	return real_t_device(powf(float(x), y));
#elif REAL_TYPE == FLOAT
	return powf(x, y);
#elif REAL_TYPE == DOUBLE
	return pow(x, y);
#endif
}

__device__   inline real_t_device sqrt_real(real_t_device x) {
#if REAL_TYPE == HALF
	return hsqrt(x);
#elif REAL_TYPE == FLOAT
	return sqrtf(x);
#elif REAL_TYPE == DOUBLE
	return sqrt(x);
#endif
}

int fread_float_to_real_t(real_t* dst, size_t siz, size_t times, FILE* fp);

#endif /* TYPE_H_ */
