CC = gcc
CFLAGS = -O3 -Wall -g -pg
LDFLAGS = -lm

all: c63enc #c63dec c63pred

%.o: %.c
	$(CC) $< $(CFLAGS) -c -o $@

c63enc: c63enc.o dsp.o tables.o io.o c63_write.o c63.h common.o me.o
	$(CC) $^ $(CFLAGS) $(LDFLAGS) -o $@
c63dec: c63dec.c dsp.o tables.o io.o c63.h common.o me.o
	$(CC) $^ $(CFLAGS) $(LDFLAGS) -o $@
c63pred: c63dec.c dsp.o tables.o io.o c63.h common.o me.o
	$(CC) $^ -DC63_PRED $(CFLAGS) $(LDFLAGS) -o $@

clean:
	rm -f *.o c63enc c63dec c63pred

encode: c63enc
	./c63enc -w 352 -h 288 -o temp/test.c63 yuv/foreman.yuv
decode:
	./c63dec temp/test.c63 yuv/test.yuv

vlc:
	vlc --rawvid-width 352 --rawvid-height 288 --rawvid-chroma I420 yuv/test.yuv
vlc-original:
	vlc --rawvid-width 352 --rawvid-height 288 --rawvid-chroma I420 yuv/foreman.yuv
vlc-reference:
	vlc --rawvid-width 352 --rawvid-height 288 --rawvid-chroma I420 yuv/reference.yuv
