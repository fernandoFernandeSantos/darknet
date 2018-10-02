/*
 * detection_gold.c
 *
 *  Created on: 02/10/2018
 *      Author: fernando
 */
#include <stdlib.h>
#include "detection_gold_w.h"
#include "detection_gold.h"
#include "darknet.h"


struct detection_gold {
	void *obj;
};

detection_gold_t* create_detection_gold(int argc, char **argv, real_t thresh,
		real_t hier_thresh, char *img_list_path, char *config_file,
		char *config_data, char *model, char *weights) {
	detection_gold_t* m;
	DetectionGold *obj;

	m = (__typeof__(m)) malloc(sizeof(*m));
	obj = new DetectionGold(argc, argv, thresh, hier_thresh, img_list_path,
			config_file, config_data, model, weights);
	m->obj = obj;
	return m;
}

void destroy_detection_gold(detection_gold_t *m, detection_gold_t* det) {
	if (m == NULL)
		return;
	delete static_cast<DetectionGold *>(m->obj);
	free (m);
}

void run(detection_gold_t *m, detection* dets, int nboxes, int img_index, int classes) {
	DetectionGold *obj;
	if (m == NULL)
		return;

	obj = static_cast<DetectionGold *>(m->obj);
	obj->run(dets, nboxes, img_index, classes);
}

void start_iteration(detection_gold_t *m) {
	DetectionGold *obj;
	if (m == NULL)
		return;

	obj = static_cast<DetectionGold *>(m->obj);
	obj->start_iteration();
}

void end_iteration(detection_gold_t *m) {
	DetectionGold *obj;
	if (m == NULL)
		return;

	obj = static_cast<DetectionGold *>(m->obj);
	obj->end_iteration();

}
