#include <assert.h>
#include <errno.h>
#include <getopt.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>

#include "dsp.h"
#include "me.h"

/* Motion estimation for 8x8 block */
static void me_block_8x8(struct c63_common *cm, int mb_x, int mb_y,
    uint8_t *orig, uint8_t *ref, int color_component)
{
  int someValue = mb_y*cm->padw[color_component]/8 + mb_x;
  struct macroblock *left_mb = &cm->curframe->mbs[color_component][someValue];
  struct macroblock *right_mb = &cm->curframe->mbs[color_component][someValue + 1];

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

  int x, y, block_row;

  int mx = mb_x * 8;
  int my = mb_y * 8;

  int best_sad_left = INT_MAX;
  int best_sad_right = INT_MAX;

  for (y = top; y < bottom; ++y)
  {
    __m128i row_sads_left1 = _mm_setzero_si128();
	__m128i row_sads_left2 = _mm_setzero_si128();

	x = left;

	if ((mb_x * 8 - left) == range)
	{
		for (block_row = 0; block_row < 8; ++block_row)
		{
			__m128i ref_pixels = _mm_loadu_si128((void const*)(ref + x + w*(block_row + y)));
			__m128i orig_pixels = _mm_loadu_si128((void const*)(orig + mx + w*(block_row + my)));

			// Left block
			row_sads_left1 = _mm_add_epi16(row_sads_left1, _mm_mpsadbw_epu8(ref_pixels, orig_pixels, B_000));
			row_sads_left2 = _mm_add_epi16(row_sads_left2, _mm_mpsadbw_epu8(ref_pixels, orig_pixels, B_101));
		}

		__m128i sad_min_and_index = _mm_minpos_epu16(_mm_add_epi16(row_sads_left1, row_sads_left2));
		int sad_min = _mm_extract_epi16(sad_min_and_index, 0);

		if (sad_min < best_sad_left)
		{
			int sad_index = _mm_extract_epi16(sad_min_and_index, 1);
			left_mb->mv_x = (x + sad_index) - mx;
			left_mb->mv_y = y - my;
			best_sad_left = sad_min;
		}

		x += 8;
	}

	__m128i row_sads_right1;
	__m128i row_sads_right2;

    for (; x < right; x+=8)
    {
    	row_sads_left1 = _mm_setzero_si128();
    	row_sads_left2 = _mm_setzero_si128();
    	__m128i row_sads_right1 = _mm_setzero_si128();
    	__m128i row_sads_right2 = _mm_setzero_si128();

		for (block_row = 0; block_row < 8; ++block_row)
		{
			__m128i ref_pixels = _mm_loadu_si128((void const*)(ref + x + w*(block_row + y)));
			__m128i orig_pixels = _mm_loadu_si128((void const*)(orig + mx + w*(block_row + my)));

			// Left block
			row_sads_left1 = _mm_add_epi16(row_sads_left1, _mm_mpsadbw_epu8(ref_pixels, orig_pixels, B_000));
			row_sads_left2 = _mm_add_epi16(row_sads_left2, _mm_mpsadbw_epu8(ref_pixels, orig_pixels, B_101));

			// Right block
			row_sads_right1 = _mm_add_epi16(row_sads_right1, _mm_mpsadbw_epu8(ref_pixels, orig_pixels, B_010));
			row_sads_right2 = _mm_add_epi16(row_sads_right2, _mm_mpsadbw_epu8(ref_pixels, orig_pixels, B_111));
		}

		__m128i sad_min_and_index = _mm_minpos_epu16(_mm_add_epi16(row_sads_left1, row_sads_left2));
		int sad_min = _mm_extract_epi16(sad_min_and_index, 0);

		if (sad_min < best_sad_left)
		{
			int sad_index = _mm_extract_epi16(sad_min_and_index, 1);
			left_mb->mv_x = (x + sad_index) - mx;
			left_mb->mv_y = y - my;
			best_sad_left = sad_min;
		}

		sad_min_and_index = _mm_minpos_epu16(_mm_add_epi16(row_sads_right1, row_sads_right2));
		sad_min = _mm_extract_epi16(sad_min_and_index, 0);

		if (sad_min < best_sad_right)
		{
			int sad_index = _mm_extract_epi16(sad_min_and_index, 1);
			right_mb->mv_x = (x + sad_index) - mx - 8;
			right_mb->mv_y = y - my;
			best_sad_right = sad_min;
		}

    }

    if ((w >= right + 16))
    {
    	row_sads_right1 = _mm_setzero_si128();
    	row_sads_right2 = _mm_setzero_si128();

    	for (block_row = 0; block_row < 8; ++block_row)
    	{
    		__m128i ref_pixels = _mm_loadu_si128((void const*)(ref + x + w*(block_row + y)));
    		__m128i orig_pixels = _mm_loadu_si128((void const*)(orig + mx + w*(block_row + my)));

    		row_sads_right1 = _mm_add_epi16(row_sads_right1, _mm_mpsadbw_epu8(ref_pixels, orig_pixels, B_010));
    		row_sads_right2 = _mm_add_epi16(row_sads_right2, _mm_mpsadbw_epu8(ref_pixels, orig_pixels, B_111));
    	}

    	__m128i sad_min_and_index = _mm_minpos_epu16(_mm_add_epi16(row_sads_right1, row_sads_right2));
		int sad_min = _mm_extract_epi16(sad_min_and_index, 0);

		if (sad_min < best_sad_right)
		{
			int sad_index = _mm_extract_epi16(sad_min_and_index, 1);
			right_mb->mv_x = (x + sad_index) - mx - 8;
			right_mb->mv_y = y - my;
			best_sad_right = sad_min;
		}
    }
  }

  /* Here, there should be a threshold on SAD that checks if the motion vector
     is cheaper than intraprediction. We always assume MV to be beneficial */

  /* printf("Using motion vector (%d, %d) with SAD %d\n", left_mb->mv_x, mb->mv_y,
     best_sad); */

  left_mb->use_mv = 1;
  right_mb->use_mv = 1;
}

void c63_motion_estimate(struct c63_common *cm)
{
  /* Compare this frame with previous reconstructed frame */
  int mb_x, mb_y;

  /* Luma */
  for (mb_y = 0; mb_y < cm->mb_rows; ++mb_y)
  {
    for (mb_x = 0; mb_x < cm->mb_cols; mb_x+=2)
    {
      me_block_8x8(cm, mb_x, mb_y, cm->curframe->orig->Y,
          cm->refframe->recons->Y, Y_COMPONENT);
    }
  }

  /* Chroma */
  for (mb_y = 0; mb_y < cm->mb_rows / 2; ++mb_y)
  {
    for (mb_x = 0; mb_x < cm->mb_cols / 2; mb_x+=2)
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
