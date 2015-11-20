#ifndef C63_INIT_H_
#define C63_INIT_H_

#include <inttypes.h>

#include "c63.h"

yuv_t* create_image(struct c63_common* cm);
void destroy_image(yuv_t* image);

struct frame* create_frame(struct c63_common *cm, yuv_t *image);
void destroy_frame(struct frame *f);

void dump_image(yuv_t *image, int w, int h, FILE *fp);

struct c63_common* init_c63_common(int width, int height);

#endif  /* C63_INIT_H_ */
