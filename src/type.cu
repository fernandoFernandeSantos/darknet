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

#if REAL_TYPE == HALF
__global__ void cuda_f32_to_f16(real_t_device* input_f32, size_t size,
		real_t_fp16 *output_f16) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
		output_f16[idx] = __float2half(input_f32[idx]);
}

__global__ void cuda_f16_to_f32(real_t_fp16* input_f16, size_t size,
		float *output_f32) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
		output_f32[idx] = __half2float(input_f16[idx]);
}

void convert_and_push_3_arrays(float *d_a, float *d_b, float *d_c,
		real_t_fp16 *a, int siz_a, real_t_fp16 *b, int siz_b, real_t_fp16 *c,
		siz_c) {

	check_error(cudaMalloc(a, sizeof(real_t_fp16) * siz_a));

	check_error(cudaMalloc(b, sizeof(real_t_fp16) * siz_b));

	check_error(cudaMalloc(c, sizeof(real_t_fp16) * siz_c));

	cuda_f32_to_f16<<<siz_a / BLOCK + 1, BLOCK>>>(d_a, siz_a, a);
	check_error(cudaPeekAtLastError());

	cuda_f32_to_f16<<<siz_b / BLOCK + 1, BLOCK>>>(d_b, siz_b, b);
	check_error(cudaPeekAtLastError());

	cuda_f32_to_f16<<<siz_c / BLOCK + 1, BLOCK>>>(d_c, siz_c, c);
	check_error(cudaPeekAtLastError());

}

void pop_and_convert_3_arrays(float *d_a, float *d_b, float *d_c,
		real_t_fp16 *a, int siz_a, real_t_fp16 *b, int siz_b, real_t_fp16 *c,
		siz_c) {
	cuda_f16_to_f32<<<siz_a / BLOCK + 1, BLOCK>>>(a, siz_a, d_a);
	check_error(cudaPeekAtLastError());

	cuda_f16_to_f32<<<siz_b / BLOCK + 1, BLOCK>>>(b, siz_b, d_b);
	check_error(cudaPeekAtLastError());

	cuda_f16_to_f32<<<siz_c / BLOCK + 1, BLOCK>>>(c, siz_c, d_c);
	check_error(cudaPeekAtLastError());

	//free the three half arrays
	check_error(cudaFree(a));

	check_error(cudaFree(b));

	check_error(cudaFree(c));

}

#endif
//
//#ifdef __NVCC__
//
//__device__                __forceinline__ real_t_device exp_real(real_t_device x) {
//#if REAL_TYPE == HALF
//	return expf(x);
//#elif REAL_TYPE == FLOAT
//	return expf(x);
//#elif REAL_TYPE == DOUBLE
//	return exp(x);
//#endif
//}
//
//__device__                __forceinline__ real_t_device floor_real(real_t_device x) {
//#if REAL_TYPE == HALF
//	return floorf(half(x));
//#elif REAL_TYPE == FLOAT
//	return floorf(x);
//#elif REAL_TYPE == DOUBLE
//	return floor(x);
//#endif
//}
//
//__device__                __forceinline__ real_t_device pow_real(real_t_device x,
//		real_t_device y) {
//#if REAL_TYPE == HALF
//	return powf(x, y);
//#elif REAL_TYPE == FLOAT
//	return powf(x, y);
//#elif REAL_TYPE == DOUBLE
//	return pow(x, y);
//#endif
//}
//
//__device__                __forceinline__ real_t_device sqrt_real(real_t_device x) {
//#if REAL_TYPE == HALF
//	return sqrtf(x);
//#elif REAL_TYPE == FLOAT
//	return sqrtf(x);
//#elif REAL_TYPE == DOUBLE
//	return sqrt(x);
//#endif
//}
//
//__device__                __forceinline__ real_t_device fabs_real(real_t_device x) {
//#if REAL_TYPE == HALF
//	return fabsf(x);
//#elif REAL_TYPE == FLOAT
//	return fabsf(x);
//#elif REAL_TYPE == DOUBLE
//	return fabs(x);
//#endif
//}
//
//__device__                __forceinline__ real_t_device log_real(real_t_device x) {
//#if REAL_TYPE == HALF
//	return hlog(x);
//#elif REAL_TYPE == FLOAT
//	return logf(x);
//#elif REAL_TYPE == DOUBLE
//	return log(x);
//#endif
//}
//
//__device__         __forceinline__ real_t_device atomic_add_real(real_t_device *x,
//		real_t_device val) {
//#if REAL_TYPE == HALF
//#if __CUDA_ARCH__ > 700
//	return atomicAdd((half*)x, (half)val);
//#endif
//
//	half old = *x;
//	*x += val;
//	return old;
//#else
//	return atomicAdd(x, val);
//#endif
//}
//
//__device__        __forceinline__ real_t_device cos_real(real_t_device x) {
//#if REAL_TYPE == HALF
//	return hcos(x);
//#elif REAL_TYPE == FLOAT
//	return cosf(x);
//#elif REAL_TYPE == DOUBLE
//	return cos(x);
//#endif
//}
//
//__device__        __forceinline__ real_t_device sin_real(real_t_device x) {
//#if REAL_TYPE == HALF
//	return hsin(x);
//#elif REAL_TYPE == FLOAT
//	return sinf(x);
//#elif REAL_TYPE == DOUBLE
//	return sin(x);
//#endif
//}

//#endif
