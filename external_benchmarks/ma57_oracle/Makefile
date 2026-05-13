# Build the MA57 oracle for the feral comparison benchmark.
# Links against the CoinHSL.v2023.11.17 bundle (same as hsl_bench).
# MA57 has no C header in this bundle; we call the Fortran MA57AD /
# MA57BD / MA57DD / MA57ID symbols directly from Fortran.

HSL_ROOT ?= $(HOME)/Dropbox/projects/CoinHSL.v2023.11.17.aarch64-apple-darwin-libgfortran5
LIBDIR   := $(HSL_ROOT)/lib

FC := gfortran
FFLAGS := -O2 -fallow-argument-mismatch -fno-aggressive-loop-optimizations

LIBS := -L$(LIBDIR) -Wl,-rpath,$(LIBDIR) \
        -lhsl -lgfortran -lopenblas -lm

ma57_bench: ma57_bench.F
	$(FC) $(FFLAGS) -o $@ $< $(LIBS)

clean:
	rm -f ma57_bench *.o

.PHONY: clean
