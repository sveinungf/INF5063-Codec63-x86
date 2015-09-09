#ifndef PRINTER_H_
#define PRINTER_H_

#include <immintrin.h>

void print_mm128i_as_uint8(const char* text, __m128i var);

void print_mm128i_as_uint16(const char* text, __m128i var);

#endif /* PRINTER_H_ */
