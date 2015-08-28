#include <assert.h>
#include <errno.h>
#include <getopt.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dsp.h"
#include "me.h"

/* Motion estimation for 8x8 block */
static void me_block_8x8(struct c63_common *cm, int mb_x, int mb_y,
    uint8_t *orig, uint8_t *ref, int color_component)
{
  struct macroblock *mb =
    &cm->curframe->mbs[color_component][mb_y*cm->padw[color_component]/8+mb_x];

  int range = cm->me_search_range;

  /* Quarter resolution for chroma channels. */
  if (color_component > 0) { range /= 2; }

  int left = mb_x * 8 - range;
  int top = mb_y * 8 - range;
  int right = mb_x * 8 + range;
  int bottom = mb_y * 8 + range;

  int w = cm->padw[color_component];
  int h = cm->padh[color_component];

  /* Make sure we are within bounds of reference frame. TODO: Support partial
     frame bounds. */
  if (left < 0) { left = 0; }
  if (top < 0) { top = 0; }
  if (right > (w - 8)) { right = w - 8; }
  if (bottom > (h - 8)) { bottom = h - 8; }

  int mx = mb_x * 8;
  int my = mb_y * 8;

  int xPattern[8] = {2, 1, 0, -1, -2, -1,  0,  1};
  int yPattern[8] = {0, 1, 2,  1,  0, -1, -2, -1};
  int xSmallPattern[4] = {1, 0, -1,  0};
  int ySmallPattern[4] = {0, 1,  0, -1};

  int origX = left + (right - left)/2;
  int origY = top + (bottom - top)/2;

  uint8_t* origBlock = orig + my*w + mx;

  int x = origX;
  int y = origY;
  int best_sad;
  sad_block_8x8(origBlock, ref + y*w+x, w, &best_sad);
  mb->mv_x = x - mx;
  mb->mv_y = y - my;

  int done = 0;

  while (!done) {
	  	int bestIndex = -1;
		int i;
		for (i = 0; i < 8; ++i)
		{
		  x = origX + xPattern[i];
		  y = origY + yPattern[i];

		  if (x > right || y > bottom) {
			  continue;
		  }

		  int sad;
		  sad_block_8x8(origBlock, ref + y*w+x, w, &sad);

		  if (sad < best_sad)
		  {
			  mb->mv_x = x - mx;
			  mb->mv_y = y - my;
			  best_sad = sad;
			  bestIndex = i;
		  }
		}

		if (bestIndex == -1) {
		  for (i = 0; i < 4; ++i) {
			  x = origX + xSmallPattern[i];
			  y = origY + ySmallPattern[i];

			  if (x > right || y > bottom) {
				  continue;
			  }

			  int sad;
			  sad_block_8x8(origBlock, ref + y*w+x, w, &sad);

			  if (sad < best_sad)
			  {
				  mb->mv_x = x - mx;
				  mb->mv_y = y - my;
				  best_sad = sad;
				  bestIndex = i;
			  }
		  }

		  done = 1;
		} else {
		  origX += xPattern[bestIndex];
		  origY += yPattern[bestIndex];
		}
  }

  /* Here, there should be a threshold on SAD that checks if the motion vector
     is cheaper than intraprediction. We always assume MV to be beneficial */

  /* printf("Using motion vector (%d, %d) with SAD %d\n", mb->mv_x, mb->mv_y,
     best_sad); */

  mb->use_mv = 1;
}

void c63_motion_estimate(struct c63_common *cm)
{
  /* Compare this frame with previous reconstructed frame */
  int mb_x, mb_y;

  /* Luma */
  for (mb_y = 0; mb_y < cm->mb_rows; ++mb_y)
  {
    for (mb_x = 0; mb_x < cm->mb_cols; ++mb_x)
    {
      me_block_8x8(cm, mb_x, mb_y, cm->curframe->orig->Y,
          cm->refframe->recons->Y, Y_COMPONENT);
    }
  }

  /* Chroma */
  for (mb_y = 0; mb_y < cm->mb_rows / 2; ++mb_y)
  {
    for (mb_x = 0; mb_x < cm->mb_cols / 2; ++mb_x)
    {
      me_block_8x8(cm, mb_x, mb_y, cm->curframe->orig->U,
          cm->refframe->recons->U, U_COMPONENT);
      me_block_8x8(cm, mb_x, mb_y, cm->curframe->orig->V,
          cm->refframe->recons->V, V_COMPONENT);
    }
  }
}

/* Motion compensation for 8x8 block */
static void mc_block_8x8(struct c63_common *cm, int mb_x, int mb_y,
    uint8_t *predicted, uint8_t *ref, int color_component)
{
  struct macroblock *mb =
    &cm->curframe->mbs[color_component][mb_y*cm->padw[color_component]/8+mb_x];

  if (!mb->use_mv) { return; }

  int left = mb_x * 8;
  int top = mb_y * 8;
  int right = left + 8;
  int bottom = top + 8;

  int w = cm->padw[color_component];

  /* Copy block from ref mandated by MV */
  int x, y;

  for (y = top; y < bottom; ++y)
  {
    for (x = left; x < right; ++x)
    {
      predicted[y*w+x] = ref[(y + mb->mv_y) * w + (x + mb->mv_x)];
    }
  }
}

void c63_motion_compensate(struct c63_common *cm)
{
  int mb_x, mb_y;

  /* Luma */
  for (mb_y = 0; mb_y < cm->mb_rows; ++mb_y)
  {
    for (mb_x = 0; mb_x < cm->mb_cols; ++mb_x)
    {
      mc_block_8x8(cm, mb_x, mb_y, cm->curframe->predicted->Y,
          cm->refframe->recons->Y, Y_COMPONENT);
    }
  }

  /* Chroma */
  for (mb_y = 0; mb_y < cm->mb_rows / 2; ++mb_y)
  {
    for (mb_x = 0; mb_x < cm->mb_cols / 2; ++mb_x)
    {
      mc_block_8x8(cm, mb_x, mb_y, cm->curframe->predicted->U,
          cm->refframe->recons->U, U_COMPONENT);
      mc_block_8x8(cm, mb_x, mb_y, cm->curframe->predicted->V,
          cm->refframe->recons->V, V_COMPONENT);
    }
  }
}
