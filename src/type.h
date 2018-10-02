#ifndef TYPE_H_
#define TYPE_H_

#define HALF 16
#define FLOAT 32
#define DOUBLE 64

#include <stdlib.h>
#include <stdio.h>

#if REAL_TYPE == HALF

#include <cuda_fp16.h>

typedef float real_t;
typedef half real_t_fp16;
//typedef real_t real_t_device;

#define FLT_MAX 1E+37

//#define REAL_INFINITY 0x7C00
#define REAL_INFINITY 0x7F800000

void transform_float_to_half_array(real_t_device* dst, float* src, size_t n);

//---------------------------------------------------------------------------------------------------

#elif REAL_TYPE == FLOAT

//FLOAT----------------------------------------------------------------------------------------------
// Single precision
typedef float real_t;
typedef real_t real_t_device;

#define FLT_MAX 1E+37

#define REAL_INFINITY 0x7F800000
//---------------------------------------------------------------------------------------------------

#elif REAL_TYPE == DOUBLE

//DOUBLE----------------------------------------------------------------------------------------------
//Double precision
typedef double real_t;
typedef real_t real_t_device;

#define FLT_MAX 1E+307

#define REAL_INFINITY 0x7FF0000000000000

//---------------------------------------------------------------------------------------------------
#endif

typedef struct __device_builtin__ {
	real_t_device x;
	real_t_device y;
	real_t_device z;
} real_t3;

#define REAL_RAND_MAX FLT_MAX

int fread_float_to_real_t(real_t* dst, size_t siz, size_t times, FILE* fp);

#if REAL_TYPE == HALF
void convert_and_push_3_arrays(float *d_a, float *d_b, float *d_c,
		real_t_fp16 *a, int siz_a, real_t_fp16 *b, int siz_b, real_t_fp16 *c,
		siz_c);

void pop_and_convert_3_arrays(float *d_a, float *d_b, float *d_c,
		real_t_fp16 *a, int siz_a, real_t_fp16 *b, int siz_b, real_t_fp16 *c,
		siz_c);
#endif
//#ifdef __NVCC__
//__device__ __forceinline__ real_t_device exp_real(real_t_device x);
//__device__ __forceinline__ real_t_device floor_real(real_t_device x);
//__device__ __forceinline__ real_t_device pow_real(real_t_device x,
//		real_t_device y);
//__device__ __forceinline__ real_t_device sqrt_real(real_t_device x);
//__device__ __forceinline__ real_t_device fabs_real(real_t_device x);
//__device__ __forceinline__ real_t_device log_real(real_t_device x);
//__device__ __forceinline__ real_t_device atomic_add_real(real_t_device *x,
//		real_t_device val);
//__device__ __forceinline__ real_t_device cos_real(real_t_device x);
//__device__ __forceinline__ real_t_device sin_real(real_t_device x);
//#endif

#endif /* TYPE_H_ */
