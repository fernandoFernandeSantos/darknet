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
#include "cuda_fp16.h"
#include "cuda_runtime.h"
#include "curand.h"

typedef __half real_t;

#define FLT_MAX 65504 - 1

#elif REAL_TYPE == FLOAT
// Single precision
typedef float real_t;

#define FLT_MAX 1E+37

#elif REAL_TYPE == DOUBLE
//Double precision
typedef double real_t;

#define FLT_MAX 1E+307

#endif

#ifdef __NVCC__

typedef struct __device_builtin__ {
	real_t x;
	real_t y;
	real_t z;
}real_t3;

#endif

int fread_float_to_real_t(real_t* dst, size_t siz, size_t times, FILE* fp);

#endif /* TYPE_H_ */
