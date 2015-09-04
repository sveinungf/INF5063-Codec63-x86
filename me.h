#ifndef C63_ME_H_
#define C63_ME_H_

#define B_000 0
#define B_001 1
#define B_010 2
#define B_011 3
#define B_100 4
#define B_101 5
#define B_110 6
#define B_111 7

#include "c63.h"
#include "printer.h"

// Declaration
void c63_motion_estimate(struct c63_common *cm);

void c63_motion_compensate(struct c63_common *cm);

#endif  /* C63_ME_H_ */
