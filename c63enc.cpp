#include <assert.h>
#include <errno.h>
#include <getopt.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "c63.h"
#include "init.h"
#include "write.h"

extern "C" {
#include "dsp.h"
#include "me.h"
#include "tables.h"
}

using namespace std;

static char *output_file, *input_file;
FILE *outfile;

static int limit_numframes = 0;

static uint32_t width;
static uint32_t height;

/* getopt */
extern int optind;
extern char *optarg;

// Get CPU cycle count
uint64_t rdtsc()
{
	unsigned int lo, hi;
	__asm__ __volatile__ ("rdtsc" : "=a" (lo), "=d" (hi));
	return ((uint64_t) hi << 32) | lo;
}

/* Read planar YUV frames with 4:2:0 chroma sub-sampling */
static bool read_yuv(FILE *file, yuv_t* image)
{
	size_t len = 0;

	/* Read Y. The size of Y is the same as the size of the image. The indices
	 represents the color component (0 is Y, 1 is U, and 2 is V) */
	len += fread(image->Y, 1, width * height, file);

	/* Read U. Given 4:2:0 chroma sub-sampling, the size is 1/4 of Y
	 because (height/2)*(width/2) = (height*width)/4. */
	len += fread(image->U, 1, (width * height) / 4, file);

	/* Read V. Given 4:2:0 chroma sub-sampling, the size is 1/4 of Y. */
	len += fread(image->V, 1, (width * height) / 4, file);

	if (ferror(file))
	{
		perror("ferror");
		exit (EXIT_FAILURE);
	}

	if (feof(file))
	{
		return false;
	}
	else if (len != width * height * 1.5)
	{
		fprintf(stderr, "Reached end of file, but incorrect bytes read.\n");
		fprintf(stderr, "Wrong input? (height: %d width: %d)\n", height, width);

		return false;
	}

	return true;
}

static void c63_encode_image(struct c63_common *cm, yuv_t *image)
{
	/* Advance to next frame */
	destroy_frame(cm->refframe);
	cm->refframe = cm->curframe;
	cm->curframe = create_frame(cm, image);

	/* Check if keyframe */
	if (cm->framenum == 0 || cm->frames_since_keyframe == cm->keyframe_interval)
	{
		cm->curframe->keyframe = 1;
		cm->frames_since_keyframe = 0;

		fprintf(stderr, " (keyframe) ");
	}
	else
	{
		cm->curframe->keyframe = 0;
	}

	if (!cm->curframe->keyframe)
	{
		/* Motion Estimation */
		c63_motion_estimate(cm, Y_COMPONENT);
		c63_motion_estimate(cm, U_COMPONENT);
		c63_motion_estimate(cm, V_COMPONENT);

		/* Motion Compensation */
		c63_motion_compensate(cm, Y_COMPONENT);
		c63_motion_compensate(cm, U_COMPONENT);
		c63_motion_compensate(cm, V_COMPONENT);
	}

	/* DCT and Quantization */
	dct_quantize(image->Y, cm->curframe->predicted->Y, cm->padw[Y_COMPONENT], cm->padh[Y_COMPONENT],
			cm->curframe->residuals->Ydct, cm->quanttbl_fp[Y_COMPONENT]);

	dct_quantize(image->U, cm->curframe->predicted->U, cm->padw[U_COMPONENT], cm->padh[U_COMPONENT],
			cm->curframe->residuals->Udct, cm->quanttbl_fp[U_COMPONENT]);

	dct_quantize(image->V, cm->curframe->predicted->V, cm->padw[V_COMPONENT], cm->padh[V_COMPONENT],
			cm->curframe->residuals->Vdct, cm->quanttbl_fp[V_COMPONENT]);

	/* Reconstruct frame for inter-prediction */
	dequantize_idct(cm->curframe->residuals->Ydct, cm->curframe->predicted->Y, cm->ypw, cm->yph,
			cm->curframe->recons->Y, cm->quanttbl_fp[Y_COMPONENT]);
	dequantize_idct(cm->curframe->residuals->Udct, cm->curframe->predicted->U, cm->upw, cm->uph,
			cm->curframe->recons->U, cm->quanttbl_fp[U_COMPONENT]);
	dequantize_idct(cm->curframe->residuals->Vdct, cm->curframe->predicted->V, cm->vpw, cm->vph,
			cm->curframe->recons->V, cm->quanttbl_fp[V_COMPONENT]);

	/* Function dump_image(), found in common.c, can be used here to check if the
	 prediction is correct */
}

static void print_help()
{
	printf("Usage: ./c63enc [options] input_file\n");
	printf("Commandline options:\n");
	printf("  -h                             Height of images to compress\n");
	printf("  -w                             Width of images to compress\n");
	printf("  -o                             Output file (.c63)\n");
	printf("  [-f]                           Limit number of frames to encode\n");
	printf("\n");

	exit (EXIT_FAILURE);
}

int main(int argc, char **argv)
{
	int c;

	if (argc == 1)
	{
		print_help();
	}

	while ((c = getopt(argc, argv, "h:w:o:f:i:")) != -1)
	{
		switch (c)
		{
			case 'h':
				height = atoi(optarg);
				break;
			case 'w':
				width = atoi(optarg);
				break;
			case 'o':
				output_file = optarg;
				break;
			case 'f':
				limit_numframes = atoi(optarg);
				break;
			default:
				print_help();
				break;
		}
	}

	if (optind >= argc)
	{
		fprintf(stderr, "Error getting program options, try --help.\n");
		exit (EXIT_FAILURE);
	}

	outfile = fopen(output_file, "wb");

	if (outfile == NULL)
	{
		perror("fopen");
		exit (EXIT_FAILURE);
	}

	struct c63_common *cm = init_c63_common(width, height);

	input_file = argv[optind];

	if (limit_numframes)
	{
		printf("Limited to %d frames.\n", limit_numframes);
	}

	FILE *infile = fopen(input_file, "rb");

	if (infile == NULL)
	{
		perror("fopen");
		exit (EXIT_FAILURE);
	}

	/* Encode input frames */
	int numframes = 0;

	yuv_t* image = create_image(cm);

# ifdef SHOW_CYCLES
	uint64_t kCycleCountTotal = 0;
# endif

	while (1)
	{
		bool ok = read_yuv(infile, image);

		if (!ok)
		{
			break;
		}

		printf("Encoding frame %d, ", numframes);

# ifdef SHOW_CYCLES
		uint64_t cycleCountBefore = rdtsc();
		c63_encode_image(cm, image);
		uint64_t cycleCountAfter = rdtsc();

		uint64_t kCycleCount = (cycleCountAfter - cycleCountBefore)/1000;
		kCycleCountTotal += kCycleCount;
		printf("%" PRIu64 "k cycles, ", kCycleCount);
# else
		c63_encode_image(cm, image);
# endif

		vector<uint8_t> byte_vector;
		write_frame_to_buffer(cm, byte_vector);
		write_buffer_to_file(byte_vector, outfile);

		++cm->framenum;
		++cm->frames_since_keyframe;

		printf("Done!\n");

		++numframes;

		if (limit_numframes && numframes >= limit_numframes)
		{
			break;
		}
	}

# ifdef SHOW_CYCLES
	printf("-----------\n");
	printf("Average CPU cycle count per frame: %" PRIu64 "k\n", kCycleCountTotal/numframes);
# endif

	destroy_image(image);

	fclose(outfile);
	fclose(infile);

	return EXIT_SUCCESS;
}
