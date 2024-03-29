#ifndef C63_DSP_SIMD_H_
#define C63_DSP_SIMD_H_

#include <inttypes.h>

#define ISQRT2 0.70710678118654f

void dct_quantize(uint8_t *in_data, uint8_t *prediction, uint32_t width, uint32_t height,
		int16_t *out_data, float *quantization);

void dequantize_idct(int16_t *in_data, uint8_t *prediction, uint32_t width, uint32_t height,
		uint8_t *out_data, float *quantization);

#endif  /* C63_DSP_SIMD_H_ */
