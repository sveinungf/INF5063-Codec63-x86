#include <inttypes.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#include "dsp.h"
#include "tables.h"

#include <immintrin.h>
#include <xmmintrin.h>

static float test2[8] __attribute__((aligned(32))) = {ISQRT2, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

static void transpose_block(float *in_data, float *out_data)
{
    int i, j;

    __m128 row1[2], row2[2], row3[2], row4[2];
    
    int k;
    for(i = 0; i < 8; i +=4 )
    {
		j = k = 0;
		
		row1[k] = _mm_load_ps(&in_data[(j*8)+i]);
		row2[k] = _mm_load_ps(&in_data[((j+1)*8)+i]);
		row3[k] = _mm_load_ps(&in_data[((j+2)*8)+i]);
		row4[k] = _mm_load_ps(&in_data[((j+3)*8)+i]);
		_MM_TRANSPOSE4_PS(row1[k], row2[k], row3[k], row4[k]);

		j += 4;
		++k;
					
		row1[k] = _mm_load_ps(&in_data[(j*8)+i]);
		row2[k] = _mm_load_ps(&in_data[((j+1)*8)+i]);
		row3[k] = _mm_load_ps(&in_data[((j+2)*8)+i]);
		row4[k] = _mm_load_ps(&in_data[((j+3)*8)+i]);
		_MM_TRANSPOSE4_PS(row1[k], row2[k], row3[k], row4[k]);
		
		_mm256_store_ps(&out_data[(i*8)], _mm256_insertf128_ps(_mm256_castps128_ps256(row1[0]), row1[1], 0b00000001));
		_mm256_store_ps(&out_data[((i+1)*8)], _mm256_insertf128_ps(_mm256_castps128_ps256(row2[0]), row2[1], 0b00000001));
		_mm256_store_ps(&out_data[((i+2)*8)], _mm256_insertf128_ps(_mm256_castps128_ps256(row3[0]), row3[1], 0b00000001));
		_mm256_store_ps(&out_data[((i+3)*8)], _mm256_insertf128_ps(_mm256_castps128_ps256(row4[0]), row4[1], 0b00000001));
		// Better to just store 128 bit at a time?	
	}
}

static void dct_1d_general(float* in_data, float* out_data, float lookup[64])
{
	__m128 in128a = _mm_load_ps(in_data);
	__m128 in128b = _mm_load_ps(in_data + 4);

	__m256 in256aa = _mm256_insertf128_ps(_mm256_castps128_ps256(in128a), in128a, 1);
	__m256 in256bb = _mm256_insertf128_ps(_mm256_castps128_ps256(in128b), in128b, 1);

	__m256 current, dct_values, multiplied, sum;

	current = _mm256_shuffle_ps(in256aa, in256aa, 0b00000000);
	dct_values = _mm256_load_ps(lookup);
	multiplied = _mm256_mul_ps(dct_values, current);
	sum = multiplied;

	current = _mm256_shuffle_ps(in256aa, in256aa, 0b01010101);
	dct_values = _mm256_load_ps(lookup + 8);
	multiplied = _mm256_mul_ps(dct_values, current);
	sum = _mm256_add_ps(sum, multiplied);

	current = _mm256_shuffle_ps(in256aa, in256aa, 0b10101010);
	dct_values = _mm256_load_ps(lookup + 16);
	multiplied = _mm256_mul_ps(dct_values, current);
	sum = _mm256_add_ps(sum, multiplied);

	current = _mm256_shuffle_ps(in256aa, in256aa, 0b11111111);
	dct_values = _mm256_load_ps(lookup + 24);
	multiplied = _mm256_mul_ps(dct_values, current);
	sum = _mm256_add_ps(sum, multiplied);

	current = _mm256_shuffle_ps(in256bb, in256bb, 0b00000000);
	dct_values = _mm256_load_ps(lookup + 32);
	multiplied = _mm256_mul_ps(dct_values, current);
	sum = _mm256_add_ps(sum, multiplied);

	current = _mm256_shuffle_ps(in256bb, in256bb, 0b01010101);
	dct_values = _mm256_load_ps(lookup + 40);
	multiplied = _mm256_mul_ps(dct_values, current);
	sum = _mm256_add_ps(sum, multiplied);

	current = _mm256_shuffle_ps(in256bb, in256bb, 0b10101010);
	dct_values = _mm256_load_ps(lookup + 48);
	multiplied = _mm256_mul_ps(dct_values, current);
	sum = _mm256_add_ps(sum, multiplied);

	current = _mm256_shuffle_ps(in256bb, in256bb, 0b11111111);
	dct_values = _mm256_load_ps(lookup + 56);
	multiplied = _mm256_mul_ps(dct_values, current);
	sum = _mm256_add_ps(sum, multiplied);

	 _mm256_store_ps(out_data, sum);
}

static void scale_block(float *in_data, float *out_data)
{
		
	__m256 v_in, v_res;
	
	static float v_au[8] __attribute__((aligned(32))) = {ISQRT2, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
	__m256 au = _mm256_load_ps((float*) &v_au);
	__m256 av = _mm256_set1_ps(ISQRT2);
	
	v_in = _mm256_load_ps(&in_data[0]);
	v_res = _mm256_mul_ps(v_in, au);
	v_res = _mm256_mul_ps(v_res, av);
	_mm256_store_ps(&out_data[0], v_res);

	v_in = _mm256_load_ps(&in_data[8]);
	v_res = _mm256_mul_ps(v_in, au);
	_mm256_store_ps(&out_data[8], v_res);
	
	v_in = _mm256_load_ps(&in_data[16]);
	v_res = _mm256_mul_ps(v_in, au);
	_mm256_store_ps(&out_data[16], v_res);
	
	
	v_in = _mm256_load_ps(&in_data[24]);
	v_res = _mm256_mul_ps(v_in, au);
	_mm256_store_ps(&out_data[24], v_res);
		
		
	v_in = _mm256_load_ps(&in_data[32]);
	v_res = _mm256_mul_ps(v_in, au);
	_mm256_store_ps(&out_data[32], v_res);
		
		
	v_in = _mm256_load_ps(&in_data[40]);
	v_res = _mm256_mul_ps(v_in, au);
	_mm256_store_ps(&out_data[40], v_res);
		
		
	v_in = _mm256_load_ps(&in_data[48]);
	v_res = _mm256_mul_ps(v_in, au);
	_mm256_store_ps(&out_data[48], v_res);
	
	v_in = _mm256_load_ps(&in_data[56]);
	v_res = _mm256_mul_ps(v_in, au);
	_mm256_store_ps(&out_data[56], v_res);
	
}

static void quantize_block(float *in_data, float *out_data, uint8_t *quant_tbl)
{
	int zigzag, i;
	
	float dct_values[8] __attribute__((aligned(32)));
	
	__m128i quants;	
	__m128 temp1, temp2;
	
	__m256 v_res, v_dct, q_tbl;
	__m256 v_temp = _mm256_set1_ps(0.25f);
	
	for (zigzag = 0; zigzag < 64; zigzag += 8)
	{
		for(i = 0; i < 8; ++i)
		{
			dct_values[i] = in_data[(zigzag_V[zigzag+i]*8) + zigzag_U[zigzag+i]];
		}
		
		quants = _mm_loadu_si128((__m128i*) &quant_tbl[zigzag]);
		
		v_dct = _mm256_load_ps((float*) &dct_values);
		v_res = _mm256_mul_ps(v_dct, v_temp);
		
		temp1 = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(quants));
		temp2 = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_shuffle_epi32(quants, 0b00000001)));
		
		q_tbl = _mm256_insertf128_ps(_mm256_castps128_ps256(temp1), temp2, 0b00000001);
		v_res = _mm256_div_ps(v_res, q_tbl);
		
		_mm256_store_ps(&out_data[zigzag], v_res);
		
		for(i = 0; i < 8; ++i)
		{
			out_data[zigzag+i] = round(out_data[zigzag+i]);
		}
	}
	/*
	for (zigzag = 0; zigzag < 64; ++zigzag)
	{
		uint8_t u = zigzag_U[zigzag];
		uint8_t v = zigzag_V[zigzag];

		float dct = in_data[v*8+u];

		// Zig-zag and quantize //
		out_data[zigzag] = (float) round((dct / 4.0) / quant_tbl[zigzag]); //_mm_cvtepi8_epi16/32
	}
	*/
}
			
static void dequantize_block(float *in_data, float *out_data,
    uint8_t *quant_tbl)
{
	int zigzag, i;
	
	float temp[8] __attribute__((aligned(32)));
	
	__m128i quants;	
	
	__m128 temp1, temp2;
	
	__m256 v_res, v_dct, q_tbl;
	__m256 v_temp = _mm256_set1_ps(0.25f);
	
	for (zigzag = 0; zigzag < 64; zigzag += 8)
	{		
		v_dct = _mm256_load_ps((float*) &in_data[zigzag]);
		quants = _mm_loadu_si128((__m128i*) &quant_tbl[zigzag]);
		
		temp1 = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(quants));
		temp2 = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_shuffle_epi32(quants, 0b00000001)));
		
		q_tbl = _mm256_insertf128_ps(_mm256_castps128_ps256(temp1), temp2, 0b00000001);
		v_res = _mm256_mul_ps(v_dct, q_tbl);
		v_res = _mm256_mul_ps(v_res, v_temp);
		_mm256_store_ps((float*) &temp, v_res);

		for(i = 0; i < 8; ++i)
		{
			out_data[(zigzag_V[zigzag+i]*8) + zigzag_U[zigzag+i]] = (float)round(temp[i]);
		}
	}
	/*
	for (zigzag = 0; zigzag < 64; ++zigzag)
	{
		uint8_t u = zigzag_U[zigzag];
		uint8_t v = zigzag_V[zigzag];

		float dct = in_data[zigzag];

		// Zig-zag and de-quantize //
		out_data[v*8+u] = (float) round((dct * quant_tbl[zigzag]) / 4.0);
	}
	*/
}


void dct_quant_block_8x8(int16_t *in_data, int16_t *out_data,
    uint8_t *quant_tbl)
{
  float mb[8*8] __attribute((aligned(32)));
  float mb2[8*8] __attribute((aligned(32)));
  
  int i, v;

  for (i = 0; i < 64; ++i) 
    {
        mb2[i] = in_data[i];
    }

  /* Two 1D DCT operations with transpose */
  for (v = 0; v < 8; ++v) 
    {
       dct_1d_general(mb2+v*8, mb+v*8, dctlookup);
    }
    
  transpose_block(mb, mb2);
  
  for (v = 0; v < 8; ++v)
    {
       dct_1d_general(mb2+v*8, mb+v*8, dctlookup);
    }
  transpose_block(mb, mb2);

  scale_block(mb2, mb);
  quantize_block(mb, mb2, quant_tbl);

  for (i = 0; i < 64; ++i) 
    {
        out_data[i] = mb2[i]; 
    }
}

void dequant_idct_block_8x8(int16_t *in_data, int16_t *out_data,
    uint8_t *quant_tbl)
{
  float mb[8*8] __attribute((aligned(32)));
  float mb2[8*8] __attribute((aligned(32)));

  int i, v;

  for (i = 0; i < 64; ++i) {
  	mb[i] = in_data[i];
  }

  dequantize_block(mb, mb2, quant_tbl);
  scale_block(mb2, mb);

  /* Two 1D inverse DCT operations with transpose */
  for (v = 0; v < 8; ++v) {
	  dct_1d_general(mb+v*8, mb2+v*8, dctlookup_trans);
  }

  transpose_block(mb2, mb);

  for (v = 0; v < 8; ++v) {
	  dct_1d_general(mb+v*8, mb2+v*8, dctlookup_trans);
  }

  transpose_block(mb2, mb);

  for (i = 0; i < 64; ++i)
  { 
	out_data[i] = mb[i];
  }
}
