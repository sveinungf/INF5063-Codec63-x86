CC = gcc

DEBUG ?= 0

VIDEO ?= 0

CCFLAGS = -Wall -march=native
LDFLAGS = -lm

<<<<<<< HEAD
all: c63enc #c63pred #c63dec
=======
ifeq ($(DEBUG),1)
	CCFLAGS += -Og -g -pg
else
	CCFLAGS += -O3
endif

ifeq ($(VIDEO),0)
	WIDTH = 352
	HEIGHT = 288
	INPUT_VIDEO = /opt/cipr/foreman.yuv
	REFERENCE_VIDEO = ~/yuv/reference/foreman.yuv
else ifeq ($(VIDEO),1)
	WIDTH = 3840
	HEIGHT = 2160
	INPUT_VIDEO = /opt/cipr/foreman_4k.yuv
	REFERENCE_VIDEO = ~/yuv/reference/foreman_4k.yuv
else ifeq ($(VIDEO),2)
	WIDTH = 1920
	HEIGHT = 1080
	INPUT_VIDEO = /opt/cipr/tractor.yuv
	REFERENCE_VIDEO = ~/yuv/reference/tractor.yuv
else ifeq ($(VIDEO),3)
	WIDTH = 4096
	HEIGHT = 1680
	INPUT_VIDEO = /opt/cipr/bagadus.yuv
endif

C63_FILE = temp/test.c63
OUTPUT_VIDEO = temp/output.yuv

all: c63enc
>>>>>>> branch 'master' of https://github.com/ikoddn/INF5063-Codec63-x86.git

%.o: %.c
	$(CC) $(CCFLAGS) -c $< -o $@

<<<<<<< HEAD
c63enc: c63enc.o dsp.o tables.o io.o c63_write.o c63.h common.o me.o
	$(CC) $^ $(CFLAGS) $(LDFLAGS) -o $@
#c63dec: c63dec.c dsp.o tables.o io.o c63.h common.o me.o
#	$(CC) $^ $(CFLAGS) $(LDFLAGS) -o $@
#c63pred: c63dec.c dsp.o tables.o io.o c63.h common.o me.o
#	$(CC) $^ -DC63_PRED $(CFLAGS) $(LDFLAGS) -o $@
=======
ALL_OBJECTS = c63_write.o c63enc.o common.o dsp.o io.o me.o tables.o

c63enc: $(ALL_OBJECTS)
	$(CC) $^ $(CCFLAGS) $(LDFLAGS) -o $@
>>>>>>> branch 'master' of https://github.com/ikoddn/INF5063-Codec63-x86.git

clean:
	rm -f c63enc $(ALL_OBJECTS) temp/*

encode: c63enc
<<<<<<< HEAD
	./c63enc -w 1920 -h 1080 -o temp/test.c63 /opt/cipr/tractor.yuv
encode2: c63enc
	./c63enc -w 352 -h 288 -o temp/test.c63 /opt/cipr/foreman.yuv
=======
	./c63enc -w $(WIDTH) -h $(HEIGHT) -o $(C63_FILE) $(INPUT_VIDEO)
>>>>>>> branch 'master' of https://github.com/ikoddn/INF5063-Codec63-x86.git
decode:
	./c63dec $(C63_FILE) $(OUTPUT_VIDEO)

vlc:
	vlc --rawvid-width $(WIDTH) --rawvid-height $(HEIGHT) --rawvid-chroma I420 $(OUTPUT_VIDEO)

gprof:
	gprof c63enc gmon.out -b

PSNR_EXEC = ./tools/yuv-tools/ycbcr.py psnr
psnr:
<<<<<<< HEAD
	./tools/yuv-tools/ycbcr.py psnr /opt/cipr/tractor.yuv 1920 1080 IYUV yuv/test.yuv
=======
	$(PSNR_EXEC) $(INPUT_VIDEO) $(WIDTH) $(HEIGHT) IYUV $(OUTPUT_VIDEO)
>>>>>>> branch 'master' of https://github.com/ikoddn/INF5063-Codec63-x86.git
psnr-reference:
<<<<<<< HEAD
	./tools/yuv-tools/ycbcr.py psnr /opt/cipr/tractor.yuv 1920 1080 IYUV /opt/cipr/tractor.yuv
=======
	$(PSNR_EXEC) $(INPUT_VIDEO) $(WIDTH) $(HEIGHT) IYUV $(REFERENCE_VIDEO)
>>>>>>> branch 'master' of https://github.com/ikoddn/INF5063-Codec63-x86.git
psnr-diff:
<<<<<<< HEAD
	./tools/yuv-tools/ycbcr.py psnr /opt/cipr/tractor.yuv 1920 1080 IYUV yuv/test.yuv
=======
	$(PSNR_EXEC) $(REFERENCE_VIDEO) $(WIDTH) $(HEIGHT) IYUV $(OUTPUT_VIDEO)
>>>>>>> branch 'master' of https://github.com/ikoddn/INF5063-Codec63-x86.git
	
cachegrind:
	valgrind --tool=cachegrind --branch-sim=yes --cachegrind-out-file=temp/cachegrind.out ./c63enc -w $(WIDTH) -h $(HEIGHT) -o $(C63_FILE) $(INPUT_VIDEO)

test: encode gprof
