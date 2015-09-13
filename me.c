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

	*result = _mm_add_epi16(row_sads1, row_sads2);
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

	*result1 = _mm_add_epi16(row_sads_left1, row_sads_left2);
	*result2 = _mm_add_epi16(row_sads_right1, row_sads_right2);
}

static void set_motion_vectors(struct macroblock* const mb, const __m128i* const min_values, const __m128i* const min_indexes, const int left, const int top, const int mx, const int my)
{
	uint16_t values[8] __attribute__((aligned(16)));
	uint16_t indexes[8] __attribute__((aligned(16)));

	_mm_store_si128((__m128i*) values, *min_values);
	_mm_store_si128((__m128i*) indexes, *min_indexes);

	unsigned int min = values[0];
	unsigned int vector_index = 0;
	unsigned int sad_index = indexes[0];

	unsigned int i;
	for (i = 1; i < 8; ++i)
	{
		if (values[i] < min)
		{
			min = values[i];
			vector_index = i;
			sad_index = indexes[i];
		}
		else if (values[i] == min && indexes[i] < sad_index)
		{
			vector_index = i;
			sad_index = indexes[i];
		}
	}

	unsigned int six = sad_index % 5;
	unsigned int siy = sad_index / 5;

	mb->mv_x = left + six*8 + vector_index - mx;
	mb->mv_y = top + siy - my;
}

/* Motion estimation for 8x8 block */
static void me_block_8x8(struct c63_common *cm, int mb_x, int mb_y,
    uint8_t *orig, uint8_t *ref, int color_component, int doleft, int doright)
{
  int mb_index = mb_y*cm->padw[color_component]/8 + mb_x;
  struct macroblock *left_mb = &cm->curframe->mbs[color_component][mb_index];
  struct macroblock *right_mb = &cm->curframe->mbs[color_component][mb_index + 1];

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

  const uint8_t* const orig_pointer = orig + mx + w*my;

  const __m128i m128i_epi16_max = _mm_set1_epi16(0x7FFF);
  const __m128i all_zeros = _mm_setzero_si128();
  const __m128i incrementor = _mm_set1_epi16(1);
  const __m128i row_incrementor = _mm_set1_epi16(5);

  __m128i counter = all_zeros;
  __m128i sad_min_values_left = m128i_epi16_max;
  __m128i sad_min_indexes_left = all_zeros;
  __m128i sad_min_values_right = m128i_epi16_max;
  __m128i sad_min_indexes_right = all_zeros;

  for (y = top; y < bottom; ++y)
  {
	  __m128i start_counter = counter;

	x = left;

	if (doleft)
	{
		uint8_t* ref_pointer = ref + x + w*y;

		__m128i next_min_left;
		sad_block_8x8(orig_pointer, ref_pointer, w, &next_min_left);

		__m128i cmpgt = _mm_cmpgt_epi16(sad_min_values_left, next_min_left);
		sad_min_values_left = _mm_min_epi16(sad_min_values_left, next_min_left);
		sad_min_indexes_left = _mm_blendv_epi8(sad_min_indexes_left, counter, cmpgt);

		x += 8;
	}

	counter = _mm_add_epi16(counter, incrementor);

    for (; x < right; x+=8)
    {
    	uint8_t* ref_pointer = ref + x + w*y;

    	__m128i next_min_left, next_min_right;
    	sad_block_2x8x8(orig_pointer, ref_pointer, w, &next_min_left, &next_min_right);

		__m128i cmpgt = _mm_cmpgt_epi16(sad_min_values_left, next_min_left);
		sad_min_values_left = _mm_min_epi16(sad_min_values_left, next_min_left);
		sad_min_indexes_left = _mm_blendv_epi8(sad_min_indexes_left, counter, cmpgt);

		cmpgt = _mm_cmpgt_epi16(sad_min_values_right, next_min_right);
		sad_min_values_right = _mm_min_epi16(sad_min_values_right, next_min_right);
		sad_min_indexes_right = _mm_blendv_epi8(sad_min_indexes_right, counter, cmpgt);

		counter = _mm_add_epi16(counter, incrementor);
    }

    if (doright)
    {
    	uint8_t* ref_pointer = ref + x + w*y;

    	__m128i next_min_right;
    	sad_block_8x8(orig_pointer + 8, ref_pointer, w, &next_min_right);

		__m128i cmpgt = _mm_cmpgt_epi16(sad_min_values_right, next_min_right);
		sad_min_values_right = _mm_min_epi16(sad_min_values_right, next_min_right);
		sad_min_indexes_right = _mm_blendv_epi8(sad_min_indexes_right, counter, cmpgt);
    }

    counter = _mm_add_epi16(start_counter, row_incrementor);
  }

  set_motion_vectors(left_mb, &sad_min_values_left, &sad_min_indexes_left, doleft ? left : left-8, top, mx, my);
  set_motion_vectors(right_mb, &sad_min_values_right, &sad_min_indexes_right, left-8, top, mx, my);

  /* Here, there should be a threshold on SAD that checks if the motion vector
       is cheaper than intraprediction. We always assume MV to be beneficial */

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
