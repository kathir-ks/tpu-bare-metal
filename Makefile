CC = gcc
CFLAGS = -Wall -O2
LDFLAGS = -ldl -lm

all: tpu_pjrt_test tpu_compute

tpu_pjrt_test: tpu_pjrt_test.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

tpu_compute: tpu_compute.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

hlo: gen_hlo.py
	python3 gen_hlo.py

clean:
	rm -f tpu_pjrt_test tpu_compute

.PHONY: all hlo clean
