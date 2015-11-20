#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

#include "io.h"

// XXX: Should be moved to a struct with FILE*

uint8_t get_byte(FILE *fp)
{
  int status = fgetc(fp);

  if (status == EOF)
  {
    fprintf(stderr, "End of file.\n");
    exit(EXIT_FAILURE);
  }

  return (uint8_t) status;
}

int read_bytes(FILE *fp, void *data, unsigned int sz)
{
  size_t status = fread(data, 1, (size_t) sz, fp);

  if ((int) status == EOF)
  {
    fprintf(stderr, "End of file.\n");
    exit(EXIT_FAILURE);
  }
  else if (status != (size_t) sz)
  {
    fprintf(stderr, "Error reading bytes\n");
    exit(EXIT_FAILURE);
  }

  return (int) status;
}

uint16_t get_bits(struct entropy_ctx *c, uint8_t n)
{
  uint16_t ret = 0;

  while(c->bit_buffer_width < n)
  {
    uint8_t b = get_byte(c->fp);
    if (b == 0xff) { get_byte(c->fp); } /* Discard stuffed byte */

    c->bit_buffer <<= 8;
    c->bit_buffer |= b;
    c->bit_buffer_width += 8;
  }

  ret = c->bit_buffer >> (c->bit_buffer_width - n);
  c->bit_buffer_width -= n;

  /* Clear grabbed bits */
  c->bit_buffer &= (1 << c->bit_buffer_width) - 1;

  return ret;
}
