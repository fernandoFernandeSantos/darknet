#include "yolo_layer.h"
#include "activations.h"
#include "blas.h"
#include "box.h"
#include "cuda.h"
#include "utils.h"

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>

layer make_yolo_layer(int batch, int w, int h, int n, int total, int *mask,
		int classes) {
	int i;
	layer l;// = { 0 };
	l.type = YOLO;

	l.n = n;
	l.total = total;
	l.batch = batch;
	l.h = h;
	l.w = w;
	l.c = n * (classes + 4 + 1);
	l.out_w = l.w;
	l.out_h = l.h;
	l.out_c = l.c;
	l.classes = classes;
	l.cost = (real_t*) calloc(1, sizeof(real_t));
	l.biases = (real_t*) calloc(total * 2, sizeof(real_t));
	if (mask)
		l.mask = mask;
	else {
		l.mask = (int*) calloc(n, sizeof(int));
		for (i = 0; i < n; ++i) {
			l.mask[i] = i;
		}
	}
	l.bias_updates = (real_t*) calloc(n * 2, sizeof(real_t));
	l.outputs = h * w * n * (classes + 4 + 1);
	l.inputs = l.outputs;
	l.truths = 90 * (4 + 1);
	l.delta =(real_t*)  calloc(batch * l.outputs, sizeof(real_t));
	l.output = (real_t*) calloc(batch * l.outputs, sizeof(real_t));
	for (i = 0; i < total * 2; ++i) {
		l.biases[i] = .5;
	}

	l.forward = forward_yolo_layer;
	l.backward = backward_yolo_layer;
#ifdef GPU
	l.forward_gpu = forward_yolo_layer_gpu;
	l.backward_gpu = backward_yolo_layer_gpu;
	l.output_gpu = cuda_make_array(l.output, batch * l.outputs);
	l.delta_gpu = cuda_make_array(l.delta, batch * l.outputs);
#endif

	fprintf(stderr, "yolo\n");
	srand(0);

	return l;
}

void resize_yolo_layer(layer *l, int w, int h) {
	l->w = w;
	l->h = h;

	l->outputs = h * w * l->n * (l->classes + 4 + 1);
	l->inputs = l->outputs;

	l->output = (real_t*) realloc(l->output, l->batch * l->outputs * sizeof(real_t));
	l->delta = (real_t*) realloc(l->delta, l->batch * l->outputs * sizeof(real_t));

#ifdef GPU
	cuda_free(l->delta_gpu);
	cuda_free(l->output_gpu);

	l->delta_gpu = cuda_make_array(l->delta, l->batch * l->outputs);
	l->output_gpu = cuda_make_array(l->output, l->batch * l->outputs);
#endif
}

box get_yolo_box(real_t *x, real_t *biases, int n, int index, int i, int j,
		int lw, int lh, int w, int h, int stride) {
	box b;
	b.x = (i + x[index + 0 * stride]) / lw;
	b.y = (j + x[index + 1 * stride]) / lh;
	b.w = exp(x[index + 2 * stride]) * biases[2 * n] / w;
	b.h = exp(x[index + 3 * stride]) * biases[2 * n + 1] / h;
	return b;
}

real_t delta_yolo_box(box truth, real_t *x, real_t *biases, int n, int index,
		int i, int j, int lw, int lh, int w, int h, real_t *delta, real_t scale,
		int stride) {
	box pred = get_yolo_box(x, biases, n, index, i, j, lw, lh, w, h, stride);
	real_t iou = box_iou(pred, truth);

	real_t tx = real_t(truth.x * lw - i);
	real_t ty = real_t(truth.y * lh - j);
	real_t tw = real_t(log(truth.w * w / biases[2 * n]));
	real_t th = real_t(log(truth.h * h / biases[2 * n + 1]));

	delta[index + 0 * stride] = scale * (tx - x[index + 0 * stride]);
	delta[index + 1 * stride] = scale * (ty - x[index + 1 * stride]);
	delta[index + 2 * stride] = scale * (tw - x[index + 2 * stride]);
	delta[index + 3 * stride] = scale * (th - x[index + 3 * stride]);
	return iou;
}

void delta_yolo_class(real_t *output, real_t *delta, int index, int class_,
		int classes, int stride, real_t *avg_cat) {
	int n;
	if (delta[index]) {
		delta[index + stride * class_] = 1 - output[index + stride * class_];
		if (avg_cat)
		*avg_cat += output[index + stride * class_];
		return;
	}
	for (n = 0; n < classes; ++n) {
		delta[index + stride * n] = ((n == class_) ? 1 : 0)
		- output[index + stride * n];
		if (n == class_ && avg_cat)
		*avg_cat += output[index + stride * n];
	}
}

static int entry_index(layer l, int batch, int location, int entry) {
	int n = location / (l.w * l.h);
	int loc = location % (l.w * l.h);
	return batch * l.outputs + n * l.w * l.h * (4 + l.classes + 1)
			+ entry * l.w * l.h + loc;
}

void forward_yolo_layer(const layer l, network net) {
	int i, j, b, t, n;
	memcpy(l.output, net.input, l.outputs * l.batch * sizeof(real_t));

#ifndef GPU
	for (b = 0; b < l.batch; ++b) {
		for (n = 0; n < l.n; ++n) {
			int index = entry_index(l, b, n * l.w * l.h, 0);
			activate_array(l.output + index, 2 * l.w * l.h, LOGISTIC);
			index = entry_index(l, b, n * l.w * l.h, 4);
			activate_array(l.output + index, (1 + l.classes) * l.w * l.h,
					LOGISTIC);
		}
	}
#endif

	memset(l.delta, 0, l.outputs * l.batch * sizeof(real_t));
	if (!net.train)
		return;
	real_t avg_iou = real_t(0);
	real_t recall = real_t(0);
	real_t recall75 = real_t(0);
	real_t avg_cat = real_t(0);
	real_t avg_obj = real_t(0);
	real_t avg_anyobj = real_t(0);
	int count = 0;
	int class_count = 0;
	*(l.cost) = 0;
	for (b = 0; b < l.batch; ++b) {
		for (j = 0; j < l.h; ++j) {
			for (i = 0; i < l.w; ++i) {
				for (n = 0; n < l.n; ++n) {
					int box_index = entry_index(l, b,
							n * l.w * l.h + j * l.w + i, 0);
					box pred = get_yolo_box(l.output, l.biases, l.mask[n],
							box_index, i, j, l.w, l.h, net.w, net.h, l.w * l.h);
					real_t best_iou = real_t(0);
					int best_t = 0;
					for (t = 0; t < l.max_boxes; ++t) {
						box truth = real_t_to_box(
								net.truth + t * (4 + 1) + b * l.truths, 1);
						if (!truth.x)
							break;
						real_t iou = box_iou(pred, truth);
						if (iou > best_iou) {
							best_iou = iou;
							best_t = t;
						}
					}
					int obj_index = entry_index(l, b,
							n * l.w * l.h + j * l.w + i, 4);
					avg_anyobj += l.output[obj_index];
					l.delta[obj_index] = 0 - l.output[obj_index];
					if (best_iou > l.ignore_thresh) {
						l.delta[obj_index] = 0;
					}
					if (best_iou > l.truth_thresh) {
						l.delta[obj_index] = 1 - l.output[obj_index];

						int class_ = net.truth[best_t * (4 + 1) + b * l.truths
						+ 4];
						if (l.map)
							class_ = l.map[class_];
						int class_index = entry_index(l, b,
								n * l.w * l.h + j * l.w + i, 4 + 1);
						delta_yolo_class(l.output, l.delta, class_index, class_,
								l.classes, l.w * l.h, 0);
						box truth = real_t_to_box(
								net.truth + best_t * (4 + 1) + b * l.truths, 1);
						delta_yolo_box(truth, l.output, l.biases, l.mask[n],
								box_index, i, j, l.w, l.h, net.w, net.h,
								l.delta, real_t(2 - truth.w * truth.h), l.w * l.h);
					}
				}
			}
		}
		for (t = 0; t < l.max_boxes; ++t) {
			box truth = real_t_to_box(net.truth + t * (4 + 1) + b * l.truths,
					1);

			if (!truth.x)
				break;
			real_t best_iou =real_t(0);
			int best_n = 0;
			i = (truth.x * l.w);
			j = (truth.y * l.h);
			box truth_shift = truth;
			truth_shift.x = truth_shift.y = 0;
			for (n = 0; n < l.total; ++n) {
				box pred = { real_t(0) };
				pred.w = l.biases[2 * n] / net.w;
				pred.h = l.biases[2 * n + 1] / net.h;
				real_t iou = box_iou(pred, truth_shift);
				if (iou > best_iou) {
					best_iou = iou;
					best_n = n;
				}
			}

			int mask_n = int_index(l.mask, best_n, l.n);
			if (mask_n >= 0) {
				int box_index = entry_index(l, b,
						mask_n * l.w * l.h + j * l.w + i, 0);
				real_t iou = delta_yolo_box(truth, l.output, l.biases, best_n,
						box_index, i, j, l.w, l.h, net.w, net.h, l.delta,
						real_t(2 - truth.w * truth.h), l.w * l.h);

				int obj_index = entry_index(l, b,
						mask_n * l.w * l.h + j * l.w + i, 4);
				avg_obj += l.output[obj_index];
				l.delta[obj_index] = 1 - l.output[obj_index];

				int class_ = net.truth[t * (4 + 1) + b * l.truths + 4];
				if (l.map)
					class_ = l.map[class_];
				int class_index = entry_index(l, b,
						mask_n * l.w * l.h + j * l.w + i, 4 + 1);
				delta_yolo_class(l.output, l.delta, class_index, class_,
						l.classes, l.w * l.h, &avg_cat);

				++count;
				++class_count;
				if (iou > .5)
					recall += 1;
				if (iou > .75)
					recall75 += 1;
				avg_iou += iou;
			}
		}
	}
	*(l.cost) = pow(mag_array(l.delta, l.outputs * l.batch), 2);
	printf(
			"Region %d Avg IOU: %f, Class: %f, Obj: %f, No Obj: %f, .5R: %f, .75R: %f,  count: %d\n",
			net.index, avg_iou / count, avg_cat / class_count, avg_obj / count,
			avg_anyobj / (l.w * l.h * l.n * l.batch), recall / count,
			recall75 / count, count);
}

void backward_yolo_layer(const layer l, network net) {
	axpy_cpu(l.batch * l.inputs,real_t(1), l.delta, 1, net.delta, 1);
}

void correct_yolo_boxes(detection *dets, int n, int w, int h, int netw,
		int neth, int relative) {
	int i;
	int new_w = 0;
	int new_h = 0;
	if (((real_t) netw / w) < ((real_t) neth / h)) {
		new_w = netw;
		new_h = (h * netw) / w;
	} else {
		new_h = neth;
		new_w = (w * neth) / h;
	}
	for (i = 0; i < n; ++i) {
		box b = dets[i].bbox;
		b.x = (b.x - (netw - new_w) / 2. / netw) / ((real_t) new_w / netw);
		b.y = (b.y - (neth - new_h) / 2. / neth) / ((real_t) new_h / neth);
		b.w *= (real_t) netw / new_w;
		b.h *= (real_t) neth / new_h;
		if (!relative) {
			b.x *= w;
			b.w *= w;
			b.y *= h;
			b.h *= h;
		}
		dets[i].bbox = b;
	}
}

int yolo_num_detections(layer l, real_t thresh) {
	int i, n;
	int count = 0;
	for (i = 0; i < l.w * l.h; ++i) {
		for (n = 0; n < l.n; ++n) {
			int obj_index = entry_index(l, 0, n * l.w * l.h + i, 4);
			if (l.output[obj_index] > thresh) {
				++count;
			}
		}
	}
	return count;
}

void avg_flipped_yolo(layer l) {
	int i, j, n, z;
	real_t *flip = l.output + l.outputs;
	for (j = 0; j < l.h; ++j) {
		for (i = 0; i < l.w / 2; ++i) {
			for (n = 0; n < l.n; ++n) {
				for (z = 0; z < l.classes + 4 + 1; ++z) {
					int i1 = z * l.w * l.h * l.n + n * l.w * l.h + j * l.w + i;
					int i2 = z * l.w * l.h * l.n + n * l.w * l.h + j * l.w
							+ (l.w - i - 1);
					real_t swap = flip[i1];
					flip[i1] = flip[i2];
					flip[i2] = swap;
					if (z == 0) {
						flip[i1] = -flip[i1];
						flip[i2] = -flip[i2];
					}
				}
			}
		}
	}
	for (i = 0; i < l.outputs; ++i) {
		l.output[i] = (l.output[i] + flip[i]) / 2.;
	}
}

int get_yolo_detections(layer l, int w, int h, int netw, int neth,
		real_t thresh, int *map, int relative, detection *dets) {
	int i, j, n;
	real_t *predictions = l.output;
	if (l.batch == 2)
		avg_flipped_yolo(l);
	int count = 0;
	for (i = 0; i < l.w * l.h; ++i) {
		int row = i / l.w;
		int col = i % l.w;
		for (n = 0; n < l.n; ++n) {
			int obj_index = entry_index(l, 0, n * l.w * l.h + i, 4);
			real_t objectness = predictions[obj_index];
			if (objectness <= thresh)
				continue;
			int box_index = entry_index(l, 0, n * l.w * l.h + i, 0);
			dets[count].bbox = get_yolo_box(predictions, l.biases, l.mask[n],
					box_index, col, row, l.w, l.h, netw, neth, l.w * l.h);
			dets[count].objectness = objectness;
			dets[count].classes = l.classes;
			for (j = 0; j < l.classes; ++j) {
				int class_index = entry_index(l, 0, n * l.w * l.h + i,
						4 + 1 + j);
				real_t prob = objectness * predictions[class_index];
				dets[count].prob[j] = (prob > thresh) ? prob : 0;
			}
			++count;
		}
	}
	correct_yolo_boxes(dets, count, w, h, netw, neth, relative);
	return count;
}

#ifdef GPU

void forward_yolo_layer_gpu(const layer l, network net) {
	copy_gpu(l.batch * l.inputs, net.input_gpu, 1, l.output_gpu, 1);
	int b, n;
	printf("passou na primeira parte da yolo layer\n");
	for (b = 0; b < l.batch; ++b) {
		for (n = 0; n < l.n; ++n) {
			int index = entry_index(l, b, n * l.w * l.h, 0);
			activate_array_gpu(l.output_gpu + index, 2 * l.w * l.h, LOGISTIC);
			index = entry_index(l, b, n * l.w * l.h, 4);
			activate_array_gpu(l.output_gpu + index,
					(1 + l.classes) * l.w * l.h, LOGISTIC);
		}
	}
	printf("passou na segunda parte da yolo layer\n");

	if (!net.train || l.onlyforward) {
		cuda_pull_array(l.output_gpu, l.output, l.batch * l.outputs);
		return;
	}
	printf("passou na terceira parte da yolo layer\n");

	cuda_pull_array(l.output_gpu, net.input, l.batch * l.inputs);
	printf("passou na quarta parte da yolo layer\n");

	forward_yolo_layer(l, net);
	printf("passou na quinta parte da yolo layer\n");

	cuda_push_array(l.delta_gpu, l.delta, l.batch * l.outputs);
	printf("passou na fechou da yolo layer\n");

}

void backward_yolo_layer_gpu(const layer l, network net) {
	axpy_gpu(l.batch * l.inputs,real_t(1), l.delta_gpu, 1, net.delta_gpu, 1);
}
#endif

