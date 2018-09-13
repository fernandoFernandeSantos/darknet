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

#if REAL_TYPE == HALF
// For half precision
#include <cuda_fp16.h>
typedef half real_t;

#elif REAL_TYPE == FLOAT
// Single precision
typedef float real_t;

#elif REAL_TYPE == DOUBLE
//Double precision
typedef double real_t;

#endif


#ifdef __NVCC__

typedef struct __device_builtin__ real_t3{
	real_t x;
	real_t y;
	real_t z;
}real_t3;

#endif

#endif /* TYPE_H_ */
