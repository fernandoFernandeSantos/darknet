#include "normalization_layer.h"
#include "blas.h"

#include <stdio.h>

layer make_normalization_layer(int batch, int w, int h, int c, int size,
		real_t alpha, real_t beta, real_t kappa) {
	fprintf(stderr,
			"Local Response Normalization Layer: %d x %d x %d image, %d size\n",
			w, h, c, size);
	layer layer; // = { 0 };
	layer.type = NORMALIZATION;
	layer.batch = batch;
	layer.h = layer.out_h = h;
	layer.w = layer.out_w = w;
	layer.c = layer.out_c = c;
	layer.kappa = kappa;
	layer.size = size;
	layer.alpha = alpha;
	layer.beta = beta;
	layer.output = (real_t*) calloc(h * w * c * batch, sizeof(real_t));
	layer.delta = (real_t*) calloc(h * w * c * batch, sizeof(real_t));
	layer.squared = (real_t*) calloc(h * w * c * batch, sizeof(real_t));
	layer.norms = (real_t*) calloc(h * w * c * batch, sizeof(real_t));
	layer.inputs = w * h * c;
	layer.outputs = layer.inputs;

	layer.forward = forward_normalization_layer;
	layer.backward = backward_normalization_layer;
#ifdef GPU
	layer.forward_gpu = forward_normalization_layer_gpu;
	layer.backward_gpu = backward_normalization_layer_gpu;

	layer.output_gpu = cuda_make_array(layer.output, h * w * c * batch);
	layer.delta_gpu = cuda_make_array(layer.delta, h * w * c * batch);
	layer.squared_gpu = cuda_make_array(layer.squared, h * w * c * batch);
	layer.norms_gpu = cuda_make_array(layer.norms, h * w * c * batch);
#endif
	return layer;
}

void resize_normalization_layer(layer *layer, int w, int h) {
	int c = layer->c;
	int batch = layer->batch;
	layer->h = h;
	layer->w = w;
	layer->out_h = h;
	layer->out_w = w;
	layer->inputs = w * h * c;
	layer->outputs = layer->inputs;
	layer->output = (real_t*) realloc(layer->output,
			h * w * c * batch * sizeof(real_t));
	layer->delta = (real_t*) realloc(layer->delta,
			h * w * c * batch * sizeof(real_t));
	layer->squared = (real_t*) realloc(layer->squared,
			h * w * c * batch * sizeof(real_t));
	layer->norms = (real_t*) realloc(layer->norms,
			h * w * c * batch * sizeof(real_t));
#ifdef GPU
	cuda_free(layer->output_gpu);
	cuda_free(layer->delta_gpu);
	cuda_free(layer->squared_gpu);
	cuda_free(layer->norms_gpu);
	layer->output_gpu = cuda_make_array(layer->output, h * w * c * batch);
	layer->delta_gpu = cuda_make_array(layer->delta, h * w * c * batch);
	layer->squared_gpu = cuda_make_array(layer->squared, h * w * c * batch);
	layer->norms_gpu = cuda_make_array(layer->norms, h * w * c * batch);
#endif
}

void forward_normalization_layer(const layer layer, network net) {
	int k, b;
	int w = layer.w;
	int h = layer.h;
	int c = layer.c;
	scal_cpu(w * h * c * layer.batch, real_t(0), layer.squared, 1);

	for (b = 0; b < layer.batch; ++b) {
		real_t *squared = layer.squared + w * h * c * b;
		real_t *norms = layer.norms + w * h * c * b;
		real_t *input = net.input + w * h * c * b;
		pow_cpu(w * h * c, real_t(2), input, 1, squared, 1);

		const_cpu(w * h, layer.kappa, norms, 1);
		for (k = 0; k < layer.size / 2; ++k) {
			axpy_cpu(w * h, layer.alpha, squared + w * h * k, 1, norms, 1);
		}

		for (k = 1; k < layer.c; ++k) {
			copy_cpu(w * h, norms + w * h * (k - 1), 1, norms + w * h * k, 1);
			int prev = k - ((layer.size - 1) / 2) - 1;
			int next = k + (layer.size / 2);
			if (prev >= 0)
				axpy_cpu(w * h, -layer.alpha, squared + w * h * prev, 1,
						norms + w * h * k, 1);
			if (next < layer.c)
				axpy_cpu(w * h, layer.alpha, squared + w * h * next, 1,
						norms + w * h * k, 1);
		}
	}
	pow_cpu(w * h * c * layer.batch, -layer.beta, layer.norms, 1, layer.output,
			1);
	mul_cpu(w * h * c * layer.batch, net.input, 1, layer.output, 1);
}

void backward_normalization_layer(const layer layer, network net) {
	// TODO This is approximate ;-)
	// Also this should add in to delta instead of overwritting.

	int w = layer.w;
	int h = layer.h;
	int c = layer.c;
	pow_cpu(w * h * c * layer.batch, -layer.beta, layer.norms, 1, net.delta, 1);
	mul_cpu(w * h * c * layer.batch, layer.delta, 1, net.delta, 1);
}

#ifdef GPU
void forward_normalization_layer_gpu(const layer layer, network net) {
	int k, b;
	int w = layer.w;
	int h = layer.h;
	int c = layer.c;
	scal_gpu(w * h * c * layer.batch, real_t(0), layer.squared_gpu, 1);

	for (b = 0; b < layer.batch; ++b) {
		real_t_device *squared = layer.squared_gpu + w * h * c * b;
		real_t_device *norms = layer.norms_gpu + w * h * c * b;
		real_t_device *input = net.input_gpu + w * h * c * b;
		pow_gpu(w * h * c, real_t(2), input, 1, squared, 1);

		const_gpu(w * h, layer.kappa, norms, 1);
		for (k = 0; k < layer.size / 2; ++k) {
			axpy_gpu(w * h, layer.alpha, squared + w * h * k, 1, norms, 1);
		}

		for (k = 1; k < layer.c; ++k) {
			copy_gpu(w * h, norms + w * h * (k - 1), 1, norms + w * h * k, 1);
			int prev = k - ((layer.size - 1) / 2) - 1;
			int next = k + (layer.size / 2);
			if (prev >= 0)
				axpy_gpu(w * h, -layer.alpha, squared + w * h * prev, 1,
						norms + w * h * k, 1);
			if (next < layer.c)
				axpy_gpu(w * h, layer.alpha, squared + w * h * next, 1,
						norms + w * h * k, 1);
		}
	}
	pow_gpu(w * h * c * layer.batch, -layer.beta, layer.norms_gpu, 1,
			layer.output_gpu, 1);
	mul_gpu(w * h * c * layer.batch, net.input_gpu, 1, layer.output_gpu, 1);
}

void backward_normalization_layer_gpu(const layer layer, network net) {
	// TODO This is approximate ;-)

	int w = layer.w;
	int h = layer.h;
	int c = layer.c;
	pow_gpu(w * h * c * layer.batch, -layer.beta, layer.norms_gpu, 1,
			net.delta_gpu, 1);
	mul_gpu(w * h * c * layer.batch, layer.delta_gpu, 1, net.delta_gpu, 1);
}
#endif
