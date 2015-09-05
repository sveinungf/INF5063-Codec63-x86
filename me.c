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
#include <stdbool.h>

#include "dsp.h"
#include "me.h"

// gcc complains if _mm_extract_epi16() has a variable as selector
static int c63_mm_extract_epi16_varselector(const __m128i a, const int imm8)
{
	switch (imm8)
	{
	case 0:
		return _mm_extract_epi16(a, 0);
	case 1:
		return _mm_extract_epi16(a, 1);
	case 2:
		return _mm_extract_epi16(a, 2);
	case 3:
		return _mm_extract_epi16(a, 3);
	case 4:
		return _mm_extract_epi16(a, 4);
	case 5:
		return _mm_extract_epi16(a, 5);
	case 6:
		return _mm_extract_epi16(a, 6);
	case 7:
		return _mm_extract_epi16(a, 7);
	default:
		return 0;
	}
}

static void sad_block_8x8(const uint8_t* const orig, const uint8_t* const ref, const int stride, __m128i* const result)
{
	const uint8_t* ref_pointer = ref;
	const uint8_t* orig_pointer = orig;

	__m128i ref_pixels = _mm_loadu_si128((void const*)(ref_pointer));
	__m128i orig_pixels = _mm_loadl_epi64((void const*)(orig_pointer));

	__m128i row_sads1 = _mm_mpsadbw_epu8(ref_pixels, orig_pixels, 0b000);
	__m128i row_sads2 = _mm_mpsadbw_epu8(ref_pixels, orig_pixels, 0b101);

	unsigned int block_row;

	// Counting down to zero creates a simpler loop termination condition
	for (block_row = 7; block_row--; )
	{
		ref_pointer += stride;
		orig_pointer += stride;

		ref_pixels = _mm_loadu_si128((void const*)(ref_pointer));
		orig_pixels = _mm_loadl_epi64((void const*)(orig_pointer));

		// Left block
		row_sads1 = _mm_add_epi16(row_sads1, _mm_mpsadbw_epu8(ref_pixels, orig_pixels, 0b000));
		row_sads2 = _mm_add_epi16(row_sads2, _mm_mpsadbw_epu8(ref_pixels, orig_pixels, 0b101));
	}

	*result = _mm_minpos_epu16(_mm_add_epi16(row_sads1, row_sads2));
}

static void sad_block_2x8x8(const uint8_t* const orig, const uint8_t* const ref, const int stride, __m128i* const result1, __m128i* const result2)
{
	const uint8_t* ref_pointer = ref;
	const uint8_t* orig_pointer = orig;

	__m128i ref_pixels = _mm_loadu_si128((void const*)(ref_pointer));
	__m128i orig_pixels = _mm_loadu_si128((void const*)(orig_pointer));

	// Left block
	__m128i row_sads_left1 = _mm_mpsadbw_epu8(ref_pixels, orig_pixels, 0b000);
	__m128i row_sads_left2 = _mm_mpsadbw_epu8(ref_pixels, orig_pixels, 0b101);

	// Right block
	__m128i row_sads_right1 = _mm_mpsadbw_epu8(ref_pixels, orig_pixels, 0b010);
	__m128i row_sads_right2 = _mm_mpsadbw_epu8(ref_pixels, orig_pixels, 0b111);

	unsigned int block_row;

	for (block_row = 7; block_row--; )
	{
		ref_pointer += stride;
		orig_pointer += stride;

		ref_pixels = _mm_loadu_si128((void const*)(ref_pointer));
		orig_pixels = _mm_loadu_si128((void const*)(orig_pointer));

		// Left block
		row_sads_left1 = _mm_add_epi16(row_sads_left1, _mm_mpsadbw_epu8(ref_pixels, orig_pixels, 0b000));
		row_sads_left2 = _mm_add_epi16(row_sads_left2, _mm_mpsadbw_epu8(ref_pixels, orig_pixels, 0b101));

		// Right block
		row_sads_right1 = _mm_add_epi16(row_sads_right1, _mm_mpsadbw_epu8(ref_pixels, orig_pixels, 0b010));
		row_sads_right2 = _mm_add_epi16(row_sads_right2, _mm_mpsadbw_epu8(ref_pixels, orig_pixels, 0b111));
	}

	*result1 = _mm_minpos_epu16(_mm_add_epi16(row_sads_left1, row_sads_left2));
	*result2 = _mm_minpos_epu16(_mm_add_epi16(row_sads_right1, row_sads_right2));
}

/* Motion estimation for 8x8 block */
static void me_block_8x8(struct c63_common *cm, int mb_x, int mb_y,
    uint8_t *orig, uint8_t *ref, int color_component, int doleft, int doright)
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

  int x, y;

  int mx = mb_x * 8;
  int my = mb_y * 8;

  int best_sad_left = INT_MAX;
  int best_sad_right = INT_MAX;

  const uint8_t* const orig_pointer = orig + mx + w*my;
  __m128i left_results[4];
  __m128i right_results[4];
  __m128i row_results_left[8];
  __m128i allOnes = _mm_set1_epi16(0xFFFF);
  int j, k;

  for (y = top; y < bottom; ++y)
  {
	  left_results[0] = allOnes;
	  left_results[1] = allOnes;
	  left_results[2] = allOnes;
	  left_results[3] = allOnes;

	x = left;
	j = 0;
	k = 0;

	if (doleft)
	{
		uint8_t* ref_pointer = ref + x + w*y;
		sad_block_8x8(orig_pointer, ref_pointer, w, &left_results[j]);
		left_results[j] = _mm_add_epi16(_mm_set_epi16(0xFFFF,0xFFFF,0xFFFF,0xFFFF,0xFFFF,0xFFFF,j*8+left-mx,0), left_results[j]);

		x += 8;
		++j;
	}

    for (; x < right; x+=8)
    {
    	uint8_t* ref_pointer = ref + x + w*y;
    	sad_block_2x8x8(orig_pointer, ref_pointer, w, &left_results[j], &right_results[k]);
    	left_results[j] = _mm_add_epi16(_mm_set_epi16(0xFFFF,0xFFFF,0xFFFF,0xFFFF,0xFFFF,0xFFFF,j*8+left-mx,0), left_results[j]);

		++j;
		++k;
    }

    if (doright)
    {
    	uint8_t* ref_pointer = ref + x + w*y;
    	sad_block_8x8(orig_pointer + 8, ref_pointer, w, &right_results[k]);

    	++k;
    }

    __m128i interleaved01 = _mm_unpacklo_epi16(left_results[0], left_results[1]);
    __m128i interleaved23 = _mm_unpacklo_epi16(left_results[2], left_results[3]);
    __m128i interleaved = _mm_unpacklo_epi32(interleaved01, interleaved23);
    __m128i minposValues = _mm_unpacklo_epi64(interleaved, allOnes);
    __m128i minposIndexes = _mm_unpackhi_epi64(interleaved, allOnes);

    __m128i result = _mm_minpos_epu16(minposValues);
    int sad_min = _mm_extract_epi16(result, 0);
    int index = _mm_extract_epi16(result, 1);
    int sad_index = c63_mm_extract_epi16_varselector(minposIndexes, index);

    row_results_left[y%8] = _mm_set_epi16(0,0,0,0,0,0,sad_index,sad_min);

    int i;

    for (i = 0; i < k; ++i)
    {
    	int sad_min = _mm_extract_epi16(right_results[i], 0);

    	if (sad_min < best_sad_right)
    	{
    		int sad_index = _mm_extract_epi16(right_results[i], 1);
    		right_mb->mv_x = left + i*8 + sad_index - mx;
    		right_mb->mv_y = y - my;
    		best_sad_right = sad_min;
    	}
    }

    if ((y+1) % 8 == 0)
    {
    	__m128i interleaved01 = _mm_unpacklo_epi16(row_results_left[0], row_results_left[1]);
    	__m128i interleaved23 = _mm_unpacklo_epi16(row_results_left[2], row_results_left[3]);
    	__m128i interleaved45 = _mm_unpacklo_epi16(row_results_left[4], row_results_left[5]);
    	__m128i interleaved67 = _mm_unpacklo_epi16(row_results_left[6], row_results_left[7]);

    	__m128i interleaved03 = _mm_unpacklo_epi32(interleaved01, interleaved23);
    	__m128i interleaved47 = _mm_unpacklo_epi32(interleaved45, interleaved67);

    	__m128i minposValues = _mm_unpacklo_epi64(interleaved03, interleaved47);
    	__m128i minposIndexes = _mm_unpackhi_epi64(interleaved03, interleaved47);

    	__m128i result = _mm_minpos_epu16(minposValues);
		int sad_min = _mm_extract_epi16(result, 0);

		if (sad_min < best_sad_left)
		{
			const int selector = _mm_extract_epi16(result, 1);
			int sad_index_x = c63_mm_extract_epi16_varselector(minposIndexes, selector);

			left_mb->mv_x = sad_index_x;
			left_mb->mv_y = selector + (y/8)*8 - my;
			best_sad_left = sad_min;
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
  uint8_t* orig_Y = cm->curframe->orig->Y;
  uint8_t* orig_U = cm->curframe->orig->U;
  uint8_t* orig_V = cm->curframe->orig->V;
  uint8_t* recons_Y = cm->refframe->recons->Y;
  uint8_t* recons_U = cm->refframe->recons->U;
  uint8_t* recons_V = cm->refframe->recons->V;

  /* Luma */
  for (mb_y = 0; mb_y < cm->mb_rows; ++mb_y)
  {
	me_block_8x8(cm, 0, mb_y, orig_Y, recons_Y, Y_COMPONENT, false, true);
	me_block_8x8(cm, 1, mb_y, orig_Y, recons_Y, Y_COMPONENT, false, true);

	unsigned int end = cm->mb_cols - 2;
    for (mb_x = 2; mb_x < end; mb_x+=2)
    {
      me_block_8x8(cm, mb_x, mb_y, orig_Y, recons_Y, Y_COMPONENT, true, true);
    }

    me_block_8x8(cm, end, mb_y, orig_Y, recons_Y, Y_COMPONENT, true, false);
    me_block_8x8(cm, end + 1, mb_y, orig_Y, recons_Y, Y_COMPONENT, true, false);
  }

  /* Chroma */
  for (mb_y = 0; mb_y < cm->mb_rows / 2; ++mb_y)
  {
	me_block_8x8(cm, 0, mb_y, orig_U, recons_U, U_COMPONENT, false, true);
	me_block_8x8(cm, 0, mb_y, orig_V, recons_V, V_COMPONENT, false, true);

	unsigned int end = (cm->mb_cols / 2) - 1;
    for (mb_x = 1; mb_x < end; mb_x+=2)
    {
      me_block_8x8(cm, mb_x, mb_y, orig_U, recons_U, U_COMPONENT, true, true);
      me_block_8x8(cm, mb_x, mb_y, orig_V, recons_V, V_COMPONENT, true, true);
    }

    me_block_8x8(cm, end, mb_y, orig_U, recons_U, U_COMPONENT, true, false);
    me_block_8x8(cm, end, mb_y, orig_V, recons_V, V_COMPONENT, true, false);
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
