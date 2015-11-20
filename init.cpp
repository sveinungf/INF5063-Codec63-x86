#include <assert.h>
#include <errno.h>
#include <getopt.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "init.h"

extern "C" {
#include "dsp.h"
#include "tables.h"
}

static const int Y = Y_COMPONENT;
static const int U = U_COMPONENT;
static const int V = V_COMPONENT;

yuv_t* create_image(struct c63_common* cm)
{
	yuv_t* image = (yuv_t*) malloc(sizeof(yuv_t));

	image->Y = (uint8_t*) malloc(cm->padw[Y] * cm->padh[Y] * sizeof(uint8_t));
	image->U = (uint8_t*) malloc(cm->padw[U] * cm->padh[U] * sizeof(uint8_t));
	image->V = (uint8_t*) malloc(cm->padw[V] * cm->padh[V] * sizeof(uint8_t));

	return image;
}

void destroy_image(yuv_t* image)
{
	free(image->Y);
	free(image->U);
	free(image->V);
	free(image);
}

void destroy_frame(struct frame *f)
{
	/* First frame doesn't have a reconstructed frame to destroy */
	if (!f)
	{
		return;
	}

	free(f->recons->Y);
	free(f->recons->U);
	free(f->recons->V);
	free(f->recons);

	free(f->residuals->Ydct);
	free(f->residuals->Udct);
	free(f->residuals->Vdct);
	free(f->residuals);

	free(f->predicted->Y);
	free(f->predicted->U);
	free(f->predicted->V);
	free(f->predicted);

	free(f->mbs[Y_COMPONENT]);
	free(f->mbs[U_COMPONENT]);
	free(f->mbs[V_COMPONENT]);

	free(f);
}

struct frame* create_frame(struct c63_common *cm, yuv_t *image)
{
	struct frame *f = (struct frame*) malloc(sizeof(struct frame));

	f->orig = image;

	f->recons = (yuv_t*) malloc(sizeof(yuv_t));
	f->recons->Y = (uint8_t*) malloc(cm->ypw * cm->yph);
	f->recons->U = (uint8_t*) malloc(cm->upw * cm->uph);
	f->recons->V = (uint8_t*) malloc(cm->vpw * cm->vph);

	f->predicted = (yuv_t*) malloc(sizeof(yuv_t));
	f->predicted->Y = (uint8_t*) calloc(cm->ypw * cm->yph, sizeof(uint8_t));
	f->predicted->U = (uint8_t*) calloc(cm->upw * cm->uph, sizeof(uint8_t));
	f->predicted->V = (uint8_t*) calloc(cm->vpw * cm->vph, sizeof(uint8_t));

	f->residuals = (dct_t*) malloc(sizeof(dct_t));
	f->residuals->Ydct = (int16_t*) calloc(cm->ypw * cm->yph, sizeof(int16_t));
	f->residuals->Udct = (int16_t*) calloc(cm->upw * cm->uph, sizeof(int16_t));
	f->residuals->Vdct = (int16_t*) calloc(cm->vpw * cm->vph, sizeof(int16_t));

	f->mbs[Y_COMPONENT] = (struct macroblock*) calloc(cm->mb_rows[Y] * cm->mb_cols[Y],
			sizeof(struct macroblock));
	f->mbs[U_COMPONENT] = (struct macroblock*) calloc(cm->mb_rows[U] * cm->mb_cols[U],
			sizeof(struct macroblock));
	f->mbs[V_COMPONENT] = (struct macroblock*) calloc(cm->mb_rows[V] * cm->mb_cols[V],
			sizeof(struct macroblock));

	return f;
}

void dump_image(yuv_t *image, int w, int h, FILE *fp)
{
	fwrite(image->Y, 1, w * h, fp);
	fwrite(image->U, 1, w * h / 4, fp);
	fwrite(image->V, 1, w * h / 4, fp);
}

struct c63_common* init_c63_common(int width, int height)
{
	int i;

	/* calloc() sets allocated memory to zero */
	struct c63_common *cm = (struct c63_common*) calloc(1, sizeof(struct c63_common));

	cm->width = width;
	cm->height = height;

	cm->padw[Y] = cm->ypw = (uint32_t) (ceil(width / 16.0f) * 16);
	cm->padh[Y] = cm->yph = (uint32_t) (ceil(height / 16.0f) * 16);
	cm->padw[U] = cm->upw = (uint32_t) (ceil(width * UX / (YX * 8.0f)) * 8);
	cm->padh[U] = cm->uph = (uint32_t) (ceil(height * UY / (YY * 8.0f)) * 8);
	cm->padw[V] = cm->vpw = (uint32_t) (ceil(width * VX / (YX * 8.0f)) * 8);
	cm->padh[V] = cm->vph = (uint32_t) (ceil(height * VY / (YY * 8.0f)) * 8);

	cm->mb_cols[Y] = cm->ypw / 8;
	cm->mb_cols[U] = cm->mb_cols[Y] / 2;
	cm->mb_cols[V] = cm->mb_cols[U];

	cm->mb_rows[Y] = cm->yph / 8;
	cm->mb_rows[U] = cm->mb_rows[Y] / 2;
	cm->mb_rows[V] = cm->mb_rows[U];

	/* Quality parameters -- Home exam deliveries should have original values,
	 i.e., quantization factor should be 25, search range should be 16, and the
	 keyframe interval should be 100. */
	cm->qp = 25;                  // Constant quantization factor. Range: [1..50]
	//cm->me_search_range = 16;   // This is now defined in c63.h
	cm->keyframe_interval = 100;  // Distance between keyframes

	/* Initialize quantization tables */
	for (i = 0; i < 64; ++i)
	{
		cm->quanttbl[Y][i] = yquanttbl_def[i] / (cm->qp / 10.0);
		cm->quanttbl[U][i] = uvquanttbl_def[i] / (cm->qp / 10.0);
		cm->quanttbl[V][i] = uvquanttbl_def[i] / (cm->qp / 10.0);

		cm->quanttbl_fp[Y_COMPONENT][i] = (uint8_t) (yquanttbl_def[i] / (cm->qp / 10.0));
		cm->quanttbl_fp[U_COMPONENT][i] = (uint8_t) (uvquanttbl_def[i] / (cm->qp / 10.0));
		cm->quanttbl_fp[V_COMPONENT][i] = (uint8_t) (uvquanttbl_def[i] / (cm->qp / 10.0));
	}

	return cm;
}
