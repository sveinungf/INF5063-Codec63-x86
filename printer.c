#include <inttypes.h>
#include <stdio.h>

#include "printer.h"

void print_mm128i_as_uint8(const char* text, __m128i var)
{
	uint8_t* val = (uint8_t*) &var;
	printf("%s", text);

	int i;
	for (i = 0; i < 16; ++i)
	{
		printf("%i ", val[i]);
	}

	puts("");
}

void print_mm128i_as_uint16(const char* text, __m128i var)
{
    uint16_t* val = (uint16_t*) &var;
    printf("%s", text);

    int i;
    for (i = 0; i < 8; ++i)
    {
    	printf("%i ", val[i]);
    }

    puts("");
}
