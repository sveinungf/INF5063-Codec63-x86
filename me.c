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

static void sad_block_8x8(const uint8_t* const orig, const uint8_t* const ref, int stride, __m128i* const result)
{
	__m128i ref_pixels = _mm_loadu_si128((void const*) ref);
	__m128i orig_pixels = _mm_loadl_epi64((void const*) orig);

	__m128i row_sads1 = _mm_mpsadbw_epu8(ref_pixels, orig_pixels, 0b000);
	__m128i row_sads2 = _mm_mpsadbw_epu8(ref_pixels, orig_pixels, 0b101);

	int offset;

	for (offset = stride; offset < 8*stride; offset += stride)
	{
		ref_pixels = _mm_loadu_si128((void const*) ref + offset);
		orig_pixels = _mm_loadl_epi64((void const*) orig + offset);

		row_sads1 = _mm_add_epi16(row_sads1, _mm_mpsadbw_epu8(ref_pixels, orig_pixels, 0b000));
		row_sads2 = _mm_add_epi16(row_sads2, _mm_mpsadbw_epu8(ref_pixels, orig_pixels, 0b101));
	}

	*result = _mm_add_epi16(row_sads1, row_sads2);
}

static void sad_block_2x8x8(const uint8_t* const orig, const uint8_t* const ref, int stride, __m128i* const result1, __m128i* const result2)
{
	__m128i ref_pixels = _mm_loadu_si128((void const*)(ref));
	__m128i orig_pixels = _mm_loadu_si128((void const*)(orig));

	// Left block
	__m128i row_sads_left1 = _mm_mpsadbw_epu8(ref_pixels, orig_pixels, 0b000);
	__m128i row_sads_left2 = _mm_mpsadbw_epu8(ref_pixels, orig_pixels, 0b101);

	// Right block
	__m128i row_sads_right1 = _mm_mpsadbw_epu8(ref_pixels, orig_pixels, 0b010);
	__m128i row_sads_right2 = _mm_mpsadbw_epu8(ref_pixels, orig_pixels, 0b111);

	int offset;

	for (offset = stride; offset < 8*stride; offset += stride)
	{
		ref_pixels = _mm_loadu_si128((void const*)(ref + offset));
		orig_pixels = _mm_loadu_si128((void const*)(orig + offset));

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

static void get_sad_index(const __m128i min_values, const __m128i min_indexes, int* sad_index_x, int* sad_index_y)
{
	uint16_t values[8] __attribute__((aligned(16)));
	uint16_t indexes[8] __attribute__((aligned(16)));

	_mm_store_si128((__m128i *) values, min_values);
	_mm_store_si128((__m128i *) indexes, min_indexes);

	int min = values[0];
	int vector_index = 0;
	int sad_index = indexes[0];

	int i;
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

	*sad_index_x = (sad_index % 5) * 8 + vector_index;
	*sad_index_y = sad_index / 5;
}

/* Motion estimation for two horizontally sequential 8x8 blocks */
static void me_block_2x8x8(struct c63_common *cm, int mb1_x, int mb_y, uint8_t *orig, uint8_t *ref, int color_component)
{
	int mb_index = mb_y * cm->padw[color_component] / 8 + mb1_x;

	// Macroblock 1
	struct macroblock *mb1 = &cm->curframe->mbs[color_component][mb_index];
	// Macroblock 2
	struct macroblock *mb2 = &cm->curframe->mbs[color_component][mb_index + 1];

	int range = cm->me_search_range;

	/* Quarter resolution for chroma channels. */
	if (color_component > 0)
	{
		range /= 2;
	}

	int m1x = mb1_x * 8;
	int m2x = m1x + 8;
	int my = mb_y * 8;

	int mb1_left = m1x - range;
	int mb2_left = m2x - range;
	int top = my - range;
	int mb1_right = m1x + range;
	int mb2_right = m2x + range;
	int bottom = my + range;

	int w = cm->padw[color_component];
	int h = cm->padh[color_component];

	/* Make sure we are within bounds of reference frame. TODO: Support partial
	 frame bounds. */
	if (mb1_left < 0)
	{
		mb1_left = mb2_left = 0;
	}
	if (top < 0)
	{
		top = 0;
	}
	if (mb1_right > (w - 8))
	{
		mb1_right = w - 8;
	}
	if (mb2_right > (w - 8))
	{
		mb2_right = w - 8;
	}
	if (bottom > (h - 8))
	{
		bottom = h - 8;
	}

	const uint8_t* const orig_pointer = orig + m1x + w * my;

	const __m128i m128i_epi16_max = _mm_set1_epi16(0x7FFF);
	const __m128i all_zeros = _mm_setzero_si128();
	const __m128i incrementor = _mm_set1_epi16(1);
	const __m128i row_incrementor = _mm_set1_epi16(5);

	__m128i counter = all_zeros;
	__m128i sad_min_values_block1 = m128i_epi16_max;
	__m128i sad_min_indexes_block1 = all_zeros;
	__m128i sad_min_values_block2 = m128i_epi16_max;
	__m128i sad_min_indexes_block2 = all_zeros;

	int y;

	for (y = top; y < bottom; ++y)
	{
		__m128i start_counter = counter;

		if (mb1_left < mb2_left)
		{
			uint8_t* ref_pointer = ref + mb1_left + w * y;

			__m128i next_min_block1;
			sad_block_8x8(orig_pointer, ref_pointer, w, &next_min_block1);

			__m128i cmpgt = _mm_cmpgt_epi16(sad_min_values_block1, next_min_block1);
			sad_min_values_block1 = _mm_min_epi16(sad_min_values_block1, next_min_block1);
			sad_min_indexes_block1 = _mm_blendv_epi8(sad_min_indexes_block1, counter, cmpgt);
		}

		counter = _mm_add_epi16(counter, incrementor);

		int x;

		for (x = mb2_left; x < mb1_right; x += 8)
		{
			uint8_t* ref_pointer = ref + x + w * y;

			__m128i next_min_block1, next_min_block2;
			sad_block_2x8x8(orig_pointer, ref_pointer, w, &next_min_block1, &next_min_block2);

			__m128i cmpgt = _mm_cmpgt_epi16(sad_min_values_block1, next_min_block1);
			sad_min_values_block1 = _mm_min_epi16(sad_min_values_block1, next_min_block1);
			sad_min_indexes_block1 = _mm_blendv_epi8(sad_min_indexes_block1, counter, cmpgt);

			cmpgt = _mm_cmpgt_epi16(sad_min_values_block2, next_min_block2);
			sad_min_values_block2 = _mm_min_epi16(sad_min_values_block2, next_min_block2);
			sad_min_indexes_block2 = _mm_blendv_epi8(sad_min_indexes_block2, counter, cmpgt);

			counter = _mm_add_epi16(counter, incrementor);
		}

		if (mb2_right > mb1_right)
		{
			uint8_t* ref_pointer = ref + mb1_right + w * y;

			__m128i next_min_block2;
			sad_block_8x8(orig_pointer + 8, ref_pointer, w, &next_min_block2);

			__m128i cmpgt = _mm_cmpgt_epi16(sad_min_values_block2, next_min_block2);
			sad_min_values_block2 = _mm_min_epi16(sad_min_values_block2, next_min_block2);
			sad_min_indexes_block2 = _mm_blendv_epi8(sad_min_indexes_block2, counter, cmpgt);
		}

		counter = _mm_add_epi16(start_counter, row_incrementor);
	}

	/* Here, there should be a threshold on SAD that checks if the motion vector
	 is cheaper than intraprediction. We always assume MV to be beneficial */

	int normalized_left = mb2_left - 8;
	int sad_index_x, sad_index_y;

	get_sad_index(sad_min_values_block1, sad_min_indexes_block1, &sad_index_x, &sad_index_y);
	mb1->mv_x = normalized_left + sad_index_x - m1x;
	mb1->mv_y = top + sad_index_y - my;
	mb1->use_mv = 1;

	get_sad_index(sad_min_values_block2, sad_min_indexes_block2, &sad_index_x, &sad_index_y);
	mb2->mv_x = normalized_left + sad_index_x - m2x;
	mb2->mv_y = top + sad_index_y - my;
	mb2->use_mv = 1;
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
		for (mb_x = 0; mb_x < cm->mb_cols; mb_x += 2)
		{
			me_block_2x8x8(cm, mb_x, mb_y, orig_Y, recons_Y, Y_COMPONENT);
		}
	}

	/* Chroma */
	for (mb_y = 0; mb_y < cm->mb_rows / 2; ++mb_y)
	{
		for (mb_x = 0; mb_x < cm->mb_cols / 2; mb_x += 2)
		{
			me_block_2x8x8(cm, mb_x, mb_y, orig_U, recons_U, U_COMPONENT);
			me_block_2x8x8(cm, mb_x, mb_y, orig_V, recons_V, V_COMPONENT);
		}
	}
}

/* Motion compensation for 8x8 block */
static void mc_block_8x8(struct c63_common *cm, int mb_x, int mb_y, uint8_t *predicted, uint8_t *ref, int color_component)
{
	struct macroblock *mb = &cm->curframe->mbs[color_component][mb_y * cm->padw[color_component] / 8 + mb_x];

	if (!mb->use_mv)
	{
		return;
	}

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
			predicted[y * w + x] = ref[(y + mb->mv_y) * w + (x + mb->mv_x)];
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
			mc_block_8x8(cm, mb_x, mb_y, cm->curframe->predicted->Y, cm->refframe->recons->Y, Y_COMPONENT);
		}
	}

	/* Chroma */
	for (mb_y = 0; mb_y < cm->mb_rows / 2; ++mb_y)
	{
		for (mb_x = 0; mb_x < cm->mb_cols / 2; ++mb_x)
		{
			mc_block_8x8(cm, mb_x, mb_y, cm->curframe->predicted->U, cm->refframe->recons->U, U_COMPONENT);
			mc_block_8x8(cm, mb_x, mb_y, cm->curframe->predicted->V, cm->refframe->recons->V, V_COMPONENT);
		}
	}
}
