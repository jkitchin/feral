#
# MUMPS 5.8.2 Makefile.inc — sequential build for macOS Apple Silicon
# Generated 2026-04-12 for the feral consensus exit benchmark.
#
# Adapted from Make.inc/Makefile.inc.generic.SEQ. Uses Homebrew GCC 15
# (gcc-15, gfortran) and Homebrew OpenBLAS (provides both BLAS and
# LAPACK in libopenblas).
#
# Build with:   cd ref/mumps && make d
# Output:       lib/libdmumps.a, lib/libmumps_common.a, libseq/libmpiseq.a,
#               PORD/lib/libpord.a
#

# Orderings: PORD only (METIS/Scotch require additional installs)
LPORDDIR = $(topdir)/PORD/lib/
IPORD    = -I$(topdir)/PORD/include/
LPORD    = -L$(LPORDDIR) -lpord$(PLAT)

ORDERINGSF  = -Dpord
ORDERINGSC  = $(ORDERINGSF)
LORDERINGS  = $(LPORD)
IORDERINGSF =
IORDERINGSC = $(IPORD)

# Library suffix and platform tag
PLAT    =
LIBEXT_SHARED = .dylib
SONAME = -install_name
SHARED_OPT = -dynamiclib
FPIC_OPT = -fPIC
LIBEXT  = .a
OUTC    = -o
OUTF    = -o
RM      = /bin/rm -f

# Compilers — use Homebrew GCC 15 to stay aligned with gfortran 15
CC      = gcc-15
FC      = gfortran
FL      = gfortran
AR      = ar vr${empty} ${empty}
RANLIB  = ranlib

# OpenBLAS (provides both BLAS and LAPACK)
OPENBLAS_DIR = /opt/homebrew/Cellar/openblas/0.3.30
LAPACK  = -L$(OPENBLAS_DIR)/lib -lopenblas
LIBBLAS = -L$(OPENBLAS_DIR)/lib -lopenblas

# Sequential MUMPS uses the bundled libmpiseq stubs
INCSEQ  = -I$(topdir)/libseq
LIBSEQ  = $(LAPACK) -L$(topdir)/libseq -lmpiseq$(PLAT)

# Pthread
LIBOTHERS = -lpthread

# Fortran/C symbol mangling: gfortran appends a single underscore
CDEFS = -DAdd_

# Compiler flags
OPTF    = -O2 -fallow-argument-mismatch -fno-aggressive-loop-optimizations
OPTC    = -O2 -I.
OPTL    = -O2

# Sequential build
INCS = $(INCSEQ)
LIBS = $(LIBSEQ)
LIBSEQNEEDED = libseqneeded
