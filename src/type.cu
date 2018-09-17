/*
 * type.c
 *
 *  Created on: 13/09/2018
 *      Author: fernando
 */

/**
 * Make real_t3 type
 * if cuda is activated it must be a function
 * accessible bye host or device
 */

#include "type.h"

int fread_float_to_real_t(real_t* dst, size_t siz, size_t times, FILE* fp) {
	float* temp = (float*) calloc(times, sizeof(float));
	if (temp == NULL) {
		free(temp);
		return -1;
	}
	int fread_result = fread(temp, sizeof(float), times, fp);
	if (fread_result != times) {
		free(temp);
		return -1;
	}
	int i;
	for (i = 0; i < times; i++) {
		//TODO: make ready for half
		dst[i] = (real_t) temp[i];
	}
	free(temp);
	return fread_result;

}