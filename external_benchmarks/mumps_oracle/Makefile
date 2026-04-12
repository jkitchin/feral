# Build the native MUMPS oracle for the feral consensus benchmark.
#
# Two-step build:
#   1. `make mumps`         — copies Makefile.inc.mumps into ref/mumps and
#                              builds the MUMPS sequential double-precision
#                              libraries (libdmumps.a, libmumps_common.a,
#                              libmpiseq.a, libpord.a). One-time, ~minutes.
#   2. `make` (default)     — builds the mumps_bench Fortran driver linked
#                              against the MUMPS libraries.
#
# `make all` does both. ref/mumps is gitignored; the Makefile.inc.mumps
# template lives here so the build configuration is version-controlled
# without leaking build artifacts.

MUMPS_DIR := ../../ref/mumps
OPENBLAS_DIR := /opt/homebrew/Cellar/openblas/0.3.30

FC := gfortran
FFLAGS := -O2 -fallow-argument-mismatch -fno-aggressive-loop-optimizations \
          -I$(MUMPS_DIR)/include -I$(MUMPS_DIR)/libseq

LIBS := -L$(MUMPS_DIR)/lib -ldmumps -lmumps_common \
        -L$(MUMPS_DIR)/PORD/lib -lpord \
        -L$(MUMPS_DIR)/libseq -lmpiseq \
        -L$(OPENBLAS_DIR)/lib -lopenblas \
        -lpthread

all: mumps mumps_bench

mumps_bench: mumps_bench.F $(MUMPS_DIR)/lib/libdmumps.a
	$(FC) $(FFLAGS) -o $@ $< $(LIBS)

# Build the MUMPS sequential libraries from ref/mumps using our template.
mumps: $(MUMPS_DIR)/lib/libdmumps.a

$(MUMPS_DIR)/lib/libdmumps.a: Makefile.inc.mumps
	cp Makefile.inc.mumps $(MUMPS_DIR)/Makefile.inc
	$(MAKE) -C $(MUMPS_DIR) d

clean:
	rm -f mumps_bench *.o

mumps-clean:
	$(MAKE) -C $(MUMPS_DIR) clean

clean-all: clean mumps-clean

.PHONY: all mumps clean mumps-clean clean-all
