#ifndef C63_IO_H_
#define C63_IO_H_

#include <inttypes.h>
#include <stdio.h>

#include "c63.h"

// Declarations

int read_bytes(FILE *fp, void *data, unsigned int sz);

uint16_t get_bits(struct entropy_ctx *c, uint8_t n);

uint8_t get_byte(FILE *fp);

#endif  /* C63_IO_H_ */
