#include <inttypes.h>
#include <math.h>
#include <stdlib.h>

#include "dsp.h"
#include "tables.h"

#include <immintrin.h>

static void transpose_block(float *in_data, float *out_data)
{
  int i, j;

  for (i = 0; i < 8; ++i)
  {
    for (j = 0; j < 8; ++j)
    {
      out_data[i*8+j] = in_data[j*8+i];
    }
  }
}

static void dct_1d(float *in_data, float *out_data)
{
  int i, j;

  for (i = 0; i < 8; ++i)
  {
    float dct = 0;

    for (j = 0; j < 8; ++j)
    {
      dct += in_data[j] * dctlookup[j][i];
    }

    out_data[i] = dct;
  }
}

static void idct_1d(float *in_data, float *out_data)
{
  int i, j;

  for (i = 0; i < 8; ++i)
  {
    float idct = 0;

    /*for (j = 0; j < 8; ++j)
    {
      idct += in_data[j] * dctlookup[i][j];
    }*/
    
    for (j = 0; j < 8; j+=4)
    {
        idct += in_data[j] * dctlookup[i][j];
        idct += in_data[j+1] * dctlookup[i][j+1];
        idct += in_data[j+2] * dctlookup[i][j+2];
        idct += in_data[j+3] * dctlookup[i][j+3];
    }
    /*
    va = _mm256_loadu_ps(in_data);
    vb = _mm256_loadu_ps(dctlookup[i]);
    vc[i] = _mm256_mul_ps(va, vb);
    
    for(j = 0; j < 8; j+=4)
    {
        idct += ((float*)&vc)[j];
        idct += ((float*)&vc)[j+1];
        idct += ((float*)&vc)[j+2];
        idct += ((float*)&vc)[j+3];
    }
    */
        
    out_data[i] = idct;
  }
}

static void scale_block(float *in_data, float *out_data)
{
  int u, v;

  for (v = 0; v < 8; ++v)
  {
    for (u = 0; u < 8; ++u)
    {
      float a1 = !u ? ISQRT2 : 1.0f;
      float a2 = !v ? ISQRT2 : 1.0f;

      /* Scale according to normalizing function */
      out_data[v*8+u] = in_data[v*8+u] * a1 * a2;
    }
  }
}

static void quantize_block(float *in_data, float *out_data, uint8_t *quant_tbl)
{
  int zigzag;

  for (zigzag = 0; zigzag < 64; ++zigzag)
  {
    uint8_t u = zigzag_U[zigzag];
    uint8_t v = zigzag_V[zigzag];

    float dct = in_data[v*8+u];

    /* Zig-zag and quantize */
    out_data[zigzag] = (float) round((dct / 4.0) / quant_tbl[zigzag]);
  }
}

static void dequantize_block(float *in_data, float *out_data,
    uint8_t *quant_tbl)
{
  int zigzag;

  for (zigzag = 0; zigzag < 64; ++zigzag)
  {
    uint8_t u = zigzag_U[zigzag];
    uint8_t v = zigzag_V[zigzag];

    float dct = in_data[zigzag];

    /* Zig-zag and de-quantize */
    out_data[v*8+u] = (float) round((dct * quant_tbl[zigzag]) / 4.0);
  }
}

void dct_quant_block_8x8(int16_t *in_data, int16_t *out_data,
    uint8_t *quant_tbl)
{
  float mb[8*8] __attribute((aligned(16)));
  float mb2[8*8] __attribute((aligned(16)));

  int i, v;

  for (i = 0; i < 64; ++i) { mb2[i] = in_data[i]; }

  /* Two 1D DCT operations with transpose */
  for (v = 0; v < 8; ++v) { dct_1d(mb2+v*8, mb+v*8); }
  transpose_block(mb, mb2);
  for (v = 0; v < 8; ++v) { dct_1d(mb2+v*8, mb+v*8); }
  transpose_block(mb, mb2);

  scale_block(mb2, mb);
  quantize_block(mb, mb2, quant_tbl);

  for (i = 0; i < 64; ++i) { out_data[i] = mb2[i]; }
}

void dequant_idct_block_8x8(int16_t *in_data, int16_t *out_data,
    uint8_t *quant_tbl)
{
  float mb[8*8] __attribute((aligned(16)));
  float mb2[8*8] __attribute((aligned(16)));

  int i, v;

  for (i = 0; i < 64; ++i) { mb[i] = in_data[i]; }

  dequantize_block(mb, mb2, quant_tbl);
  scale_block(mb2, mb);

  /* Two 1D inverse DCT operations with transpose */
  for (v = 0; v < 8; ++v) { idct_1d(mb+v*8, mb2+v*8); }
  transpose_block(mb2, mb);
  for (v = 0; v < 8; ++v) { idct_1d(mb+v*8, mb2+v*8); }
  transpose_block(mb2, mb);

  for (i = 0; i < 64; ++i) { out_data[i] = mb[i]; }
}
