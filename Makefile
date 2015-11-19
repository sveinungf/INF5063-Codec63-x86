CC = gcc
CFLAGS = -march=native -O3 -Wall -DSHOW_CYCLES -g -pg
LDFLAGS = -lm

all: c63enc #c63pred #c63dec

%.o: %.c
	$(CC) $< $(CFLAGS) -c -o $@

c63enc: c63enc.o dsp.o tables.o io.o c63_write.o c63.h common.o me.o
	$(CC) $^ $(CFLAGS) $(LDFLAGS) -o $@
#c63dec: c63dec.c dsp.o tables.o io.o c63.h common.o me.o
#	$(CC) $^ $(CFLAGS) $(LDFLAGS) -o $@
#c63pred: c63dec.c dsp.o tables.o io.o c63.h common.o me.o
#	$(CC) $^ -DC63_PRED $(CFLAGS) $(LDFLAGS) -o $@

clean:
	rm -f *.o c63enc temp/* yuv/test.yuv c63pred #c63dec

encode: c63enc
	./c63enc -w 1920 -h 1080 -o temp/test.c63 /opt/cipr/tractor.yuv
encode2: c63enc
	./c63enc -w 352 -h 288 -o temp/test.c63 /opt/cipr/foreman.yuv
decode:
	./c63dec temp/test.c63 yuv/test.yuv

vlc:
	vlc --rawvid-width 352 --rawvid-height 288 --rawvid-chroma I420 yuv/test.yuv
vlc-original:
	vlc --rawvid-width 352 --rawvid-height 288 --rawvid-chroma I420 yuv/foreman.yuv
vlc-reference:
	vlc --rawvid-width 352 --rawvid-height 288 --rawvid-chroma I420 yuv/reference.yuv

gprof:
	gprof c63enc gmon.out -b
gprof-file:
	gprof c63enc gmon.out > temp/gprof-result.txt

psnr:
	./tools/yuv-tools/ycbcr.py psnr /opt/cipr/tractor.yuv 1920 1080 IYUV yuv/test.yuv
psnr-reference:
	./tools/yuv-tools/ycbcr.py psnr /opt/cipr/tractor.yuv 1920 1080 IYUV /opt/cipr/tractor.yuv
psnr-diff:
	./tools/yuv-tools/ycbcr.py psnr /opt/cipr/tractor.yuv 1920 1080 IYUV yuv/test.yuv
	
cachegrind:
	valgrind --tool=cachegrind --branch-sim=yes --cachegrind-out-file=temp/cachegrind.out ./c63enc -w 352 -h 288 -f 30 -o temp/test.c63 yuv/foreman.yuv

test: encode gprof
