#include "connected_layer.h"
#include "convolutional_layer.h"
#include "batchnorm_layer.h"
#include "utils.h"
#include "cuda.h"
#include "blas.h"
#include "gemm.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

layer make_connected_layer(int batch, int inputs, int outputs,
		ACTIVATION activation, int batch_normalize, int adam) {
	int i;
	layer l; // = { 0 };
	l.learning_rate_scale = 1;
	l.type = CONNECTED;

	l.inputs = inputs;
	l.outputs = outputs;
	l.batch = batch;
	l.batch_normalize = batch_normalize;
	l.h = 1;
	l.w = 1;
	l.c = inputs;
	l.out_h = 1;
	l.out_w = 1;
	l.out_c = outputs;

	l.output = (real_t*) calloc(batch * outputs, sizeof(real_t));
	l.delta = (real_t*) calloc(batch * outputs, sizeof(real_t));

	l.weight_updates = (real_t*) calloc(inputs * outputs, sizeof(real_t));
	l.bias_updates = (real_t*) calloc(outputs, sizeof(real_t));

	l.weights = (real_t*) calloc(outputs * inputs, sizeof(real_t));
	l.biases = (real_t*) calloc(outputs, sizeof(real_t));

	l.forward = forward_connected_layer;
	l.backward = backward_connected_layer;
	l.update = update_connected_layer;

	//real_t scale = 1./sqrt(inputs);
	real_t scale = real_t(sqrt(2. / inputs));
	for (i = 0; i < outputs * inputs; ++i) {
		l.weights[i] = scale * rand_uniform(real_t(-1), real_t(1));
	}

	for (i = 0; i < outputs; ++i) {
		l.biases[i] = 0;
	}

	if (adam) {
		l.m = (real_t*) calloc(l.inputs * l.outputs, sizeof(real_t));
		l.v = (real_t*) calloc(l.inputs * l.outputs, sizeof(real_t));
		l.bias_m = (real_t*) calloc(l.outputs, sizeof(real_t));
		l.scale_m = (real_t*) calloc(l.outputs, sizeof(real_t));
		l.bias_v = (real_t*) calloc(l.outputs, sizeof(real_t));
		l.scale_v = (real_t*) calloc(l.outputs, sizeof(real_t));
	}
	if (batch_normalize) {
		l.scales = (real_t*) calloc(outputs, sizeof(real_t));
		l.scale_updates = (real_t*) calloc(outputs, sizeof(real_t));
		for (i = 0; i < outputs; ++i) {
			l.scales[i] = 1;
		}

		l.mean = (real_t*) calloc(outputs, sizeof(real_t));
		l.mean_delta = (real_t*) calloc(outputs, sizeof(real_t));
		l.variance = (real_t*) calloc(outputs, sizeof(real_t));
		l.variance_delta = (real_t*) calloc(outputs, sizeof(real_t));

		l.rolling_mean = (real_t*) calloc(outputs, sizeof(real_t));
		l.rolling_variance = (real_t*) calloc(outputs, sizeof(real_t));

		l.x = (real_t*) calloc(batch * outputs, sizeof(real_t));
		l.x_norm = (real_t*) calloc(batch * outputs, sizeof(real_t));
	}

#ifdef GPU
	l.forward_gpu = forward_connected_layer_gpu;
	l.backward_gpu = backward_connected_layer_gpu;
	l.update_gpu = update_connected_layer_gpu;

	l.weights_gpu = cuda_make_array(l.weights, outputs * inputs);
	l.biases_gpu = cuda_make_array(l.biases, outputs);

	l.weight_updates_gpu = cuda_make_array(l.weight_updates, outputs * inputs);
	l.bias_updates_gpu = cuda_make_array(l.bias_updates, outputs);

	l.output_gpu = cuda_make_array(l.output, outputs * batch);
	l.delta_gpu = cuda_make_array(l.delta, outputs * batch);
	if (adam) {
		l.m_gpu = cuda_make_array(0, inputs * outputs);
		l.v_gpu = cuda_make_array(0, inputs * outputs);
		l.bias_m_gpu = cuda_make_array(0, outputs);
		l.bias_v_gpu = cuda_make_array(0, outputs);
		l.scale_m_gpu = cuda_make_array(0, outputs);
		l.scale_v_gpu = cuda_make_array(0, outputs);
	}

	if (batch_normalize) {
		l.mean_gpu = cuda_make_array(l.mean, outputs);
		l.variance_gpu = cuda_make_array(l.variance, outputs);

		l.rolling_mean_gpu = cuda_make_array(l.mean, outputs);
		l.rolling_variance_gpu = cuda_make_array(l.variance, outputs);

		l.mean_delta_gpu = cuda_make_array(l.mean, outputs);
		l.variance_delta_gpu = cuda_make_array(l.variance, outputs);

		l.scales_gpu = cuda_make_array(l.scales, outputs);
		l.scale_updates_gpu = cuda_make_array(l.scale_updates, outputs);

		l.x_gpu = cuda_make_array(l.output, l.batch * outputs);
		l.x_norm_gpu = cuda_make_array(l.output, l.batch * outputs);
#ifdef CUDNN
		cudnnCreateTensorDescriptor(&l.normTensorDesc);
		cudnnCreateTensorDescriptor(&l.dstTensorDesc);
		cudnnSetTensor4dDescriptor(l.dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l.batch, l.out_c, l.out_h, l.out_w);
		cudnnSetTensor4dDescriptor(l.normTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, l.out_c, 1, 1);
#endif
	}
#endif
	l.activation = activation;
	fprintf(stderr, "connected                            %4d  ->  %4d\n",
			inputs, outputs);
	return l;
}

void update_connected_layer(layer l, update_args a) {
	real_t learning_rate = a.learning_rate * l.learning_rate_scale;
	real_t momentum = a.momentum;
	real_t decay = a.decay;
	int batch = a.batch;
	axpy_cpu(l.outputs, real_t(learning_rate / batch), l.bias_updates, 1, l.biases, 1);
	scal_cpu(l.outputs, momentum, l.bias_updates, 1);

	if (l.batch_normalize) {
		axpy_cpu(l.outputs, real_t(learning_rate / batch), l.scale_updates, 1, l.scales,
				1);
		scal_cpu(l.outputs, momentum, l.scale_updates, 1);
	}

	axpy_cpu(l.inputs * l.outputs, real_t(-decay * batch), l.weights, 1,
			l.weight_updates, 1);
	axpy_cpu(l.inputs * l.outputs, real_t(learning_rate / batch), l.weight_updates, 1,
			l.weights, 1);
	scal_cpu(l.inputs * l.outputs, momentum, l.weight_updates, 1);
}

void forward_connected_layer(layer l, network net) {
	fill_cpu(l.outputs * l.batch, real_t(0), l.output, 1);
	int m = l.batch;
	int k = l.inputs;
	int n = l.outputs;
	real_t *a = net.input;
	real_t *b = l.weights;
	real_t *c = l.output;
	gemm(0, 1, m, n, k, real_t(1), a, k, b, k, real_t(1), c, n);
	if (l.batch_normalize) {
		forward_batchnorm_layer(l, net);
	} else {
		add_bias(l.output, l.biases, l.batch, l.outputs, 1);
	}
	activate_array(l.output, l.outputs * l.batch, l.activation);
}

void backward_connected_layer(layer l, network net) {
	gradient_array(l.output, l.outputs * l.batch, l.activation, l.delta);

	if (l.batch_normalize) {
		backward_batchnorm_layer(l, net);
	} else {
		backward_bias(l.bias_updates, l.delta, l.batch, l.outputs, 1);
	}

	int m = l.outputs;
	int k = l.batch;
	int n = l.inputs;
	real_t *a = l.delta;
	real_t *b = net.input;
	real_t *c = l.weight_updates;
	gemm(1, 0, m, n, k, real_t(1), a, m, b, n, real_t(1), c, n);

	m = l.batch;
	k = l.outputs;
	n = l.inputs;

	a = l.delta;
	b = l.weights;
	c = net.delta;

	if (c)
		gemm(0, 0, m, n, k, real_t(1), a, k, b, n, real_t(1), c, n);
}

void denormalize_connected_layer(layer l) {
	int i, j;
	for (i = 0; i < l.outputs; ++i) {
		real_t scale = real_t(l.scales[i] / sqrt(l.rolling_variance[i] + .000001));
		for (j = 0; j < l.inputs; ++j) {
			l.weights[i * l.inputs + j] *= scale;
		}
		l.biases[i] -= l.rolling_mean[i] * scale;
		l.scales[i] = 1;
		l.rolling_mean[i] = 0;
		l.rolling_variance[i] = 1;
	}
}

void statistics_connected_layer(layer l) {
	if (l.batch_normalize) {
		printf("Scales ");
		print_statistics(l.scales, l.outputs);
		/*
		 printf("Rolling Mean ");
		 print_statistics(l.rolling_mean, l.outputs);
		 printf("Rolling Variance ");
		 print_statistics(l.rolling_variance, l.outputs);
		 */
	}
	printf("Biases ");
	print_statistics(l.biases, l.outputs);
	printf("Weights ");
	print_statistics(l.weights, l.outputs);
}

#ifdef GPU

void pull_connected_layer(layer l) {
	cuda_pull_array(l.weights_gpu, l.weights, l.inputs * l.outputs);
	cuda_pull_array(l.biases_gpu, l.biases, l.outputs);
	cuda_pull_array(l.weight_updates_gpu, l.weight_updates,
			l.inputs * l.outputs);
	cuda_pull_array(l.bias_updates_gpu, l.bias_updates, l.outputs);
	if (l.batch_normalize) {
		cuda_pull_array(l.scales_gpu, l.scales, l.outputs);
		cuda_pull_array(l.rolling_mean_gpu, l.rolling_mean, l.outputs);
		cuda_pull_array(l.rolling_variance_gpu, l.rolling_variance, l.outputs);
	}
}

void push_connected_layer(layer l) {
	cuda_push_array(l.weights_gpu, l.weights, l.inputs * l.outputs);
	cuda_push_array(l.biases_gpu, l.biases, l.outputs);
	cuda_push_array(l.weight_updates_gpu, l.weight_updates,
			l.inputs * l.outputs);
	cuda_push_array(l.bias_updates_gpu, l.bias_updates, l.outputs);
	if (l.batch_normalize) {
		cuda_push_array(l.scales_gpu, l.scales, l.outputs);
		cuda_push_array(l.rolling_mean_gpu, l.rolling_mean, l.outputs);
		cuda_push_array(l.rolling_variance_gpu, l.rolling_variance, l.outputs);
	}
}

void update_connected_layer_gpu(layer l, update_args a) {
	real_t learning_rate = a.learning_rate * l.learning_rate_scale;
	real_t momentum = a.momentum;
	real_t decay = a.decay;
	int batch = a.batch;
	if (a.adam) {
		adam_update_gpu(l.weights_gpu, l.weight_updates_gpu, l.m_gpu, l.v_gpu,
				CAST(a.B1), CAST(a.B2), CAST(a.eps), CAST(decay), CAST(learning_rate), l.inputs * l.outputs,
				batch, a.t);
		adam_update_gpu(l.biases_gpu, l.bias_updates_gpu, l.bias_m_gpu,
				l.bias_v_gpu, CAST(a.B1), CAST(a.B2), CAST(a.eps), CAST(decay), CAST(learning_rate),
				l.outputs, batch, a.t);
		if (l.scales_gpu) {
			adam_update_gpu(l.scales_gpu, l.scale_updates_gpu, l.scale_m_gpu,
					l.scale_v_gpu, CAST(a.B1), CAST(a.B2), CAST(a.eps), CAST(decay), CAST(learning_rate),
					l.outputs, batch, a.t);
		}
	} else {
		axpy_gpu(l.outputs, (learning_rate / batch), l.bias_updates_gpu, 1,
				l.biases_gpu, 1);
		scal_gpu(l.outputs, CAST(momentum), l.bias_updates_gpu, 1);

		if (l.batch_normalize) {
			axpy_gpu(l.outputs, (learning_rate / batch), l.scale_updates_gpu, 1,
					l.scales_gpu, 1);
			scal_gpu(l.outputs, CAST(momentum), l.scale_updates_gpu, 1);
		}

		axpy_gpu(l.inputs * l.outputs, (-decay * batch), l.weights_gpu, 1,
				l.weight_updates_gpu, 1);
		axpy_gpu(l.inputs * l.outputs, (learning_rate / batch),
				l.weight_updates_gpu, 1, l.weights_gpu, 1);
		scal_gpu(l.inputs * l.outputs, CAST(momentum), l.weight_updates_gpu, 1);
	}
}

void forward_connected_layer_gpu(layer l, network net) {
	fill_gpu(l.outputs * l.batch, (0), l.output_gpu, 1);

	int m = l.batch;
	int k = l.inputs;
	int n = l.outputs;
	real_t_device * a = net.input_gpu;
	real_t_device * b = l.weights_gpu;
	real_t_device * c = l.output_gpu;
	gemm_gpu(0, 1, m, n, k, (1), a, k, b, k, (1), c, n);

	if (l.batch_normalize) {
		forward_batchnorm_layer_gpu(l, net);
	} else {
		add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.outputs, 1);
	}
	activate_array_gpu(l.output_gpu, l.outputs * l.batch, l.activation);
}

void backward_connected_layer_gpu(layer l, network net) {
	constrain_gpu(l.outputs * l.batch, (1), l.delta_gpu, 1);
	gradient_array_gpu(l.output_gpu, l.outputs * l.batch, l.activation,
			l.delta_gpu);
	if (l.batch_normalize) {
		backward_batchnorm_layer_gpu(l, net);
	} else {
		backward_bias_gpu(l.bias_updates_gpu, l.delta_gpu, l.batch, l.outputs,
				1);
	}

	int m = l.outputs;
	int k = l.batch;
	int n = l.inputs;
	real_t_device * a = l.delta_gpu;
	real_t_device * b = net.input_gpu;
	real_t_device * c = l.weight_updates_gpu;
	gemm_gpu(1, 0, m, n, k, (1), a, m, b, n, (1), c, n);

	m = l.batch;
	k = l.outputs;
	n = l.inputs;

	a = l.delta_gpu;
	b = l.weights_gpu;
	c = net.delta_gpu;

	if (c)
		gemm_gpu(0, 0, m, n, k, (1), a, k, b, n, (1), c, n);
}
#endif
