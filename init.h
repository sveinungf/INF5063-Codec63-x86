#ifndef C63_INIT_H_
#define C63_INIT_H_

#include <inttypes.h>

#include "c63.h"

yuv_t* create_image(struct c63_common* cm);
void destroy_image(yuv_t* image);

void dump_image(yuv_t *image, int w, int h, FILE *fp);

struct c63_common* init_c63_common(int width, int height);
void cleanup_c63_common(struct c63_common* cm);

#endif  /* C63_INIT_H_ */
