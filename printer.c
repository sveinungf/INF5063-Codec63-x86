#include "printer.h"

void print128_num(__m128i var)
{
    uint16_t *val = (uint16_t*) &var;
    printf("Numerical: %i %i %i %i %i %i %i %i \n",
           val[0], val[1], val[2], val[3], val[4], val[5],
           val[6], val[7]);
}
