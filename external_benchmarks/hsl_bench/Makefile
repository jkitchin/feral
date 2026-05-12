# Build hsl_bench against the CoinHSL.v2023.11.17 bundle in Dropbox.
# Pure C; links libhsl which pulls in libgfortran + libopenblas.

HSL_ROOT ?= $(HOME)/Dropbox/projects/CoinHSL.v2023.11.17.aarch64-apple-darwin-libgfortran5
INCDIR   := $(HSL_ROOT)/include
LIBDIR   := $(HSL_ROOT)/lib

CFLAGS  := -O2 -Wall -Wextra -std=c11 -I$(INCDIR)
LDFLAGS := -L$(LIBDIR) -Wl,-rpath,$(LIBDIR)
LIBS    := -lhsl -lgfortran -lopenblas -lm

hsl_bench: hsl_bench.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS) $(LIBS)

clean:
	rm -f hsl_bench

.PHONY: clean
