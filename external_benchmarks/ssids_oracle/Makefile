# Build the SSIDS oracle for the feral consensus benchmark.
#
# Two-step build:
#   1. Build SPRAL with meson (one-time).
#   2. Build the ssids_bench Fortran driver linked against libspral.
#
# Run with OMP_CANCELLATION=true (SSIDS requires it).

SPRAL_DIR := ../../ref/spral
SPRAL_BUILD := $(SPRAL_DIR)/build

FC := gfortran
FFLAGS := -O2 -fallow-argument-mismatch \
          -I$(SPRAL_BUILD)/libspral.dylib.p \
          -I$(SPRAL_BUILD)

# Two rpaths for libspral.dylib:
#   1. @loader_path-relative — survives a repo move without a rebuild
#      (the binary is at external_benchmarks/ssids_oracle/, libspral is
#      at ref/spral/build/, i.e. two levels up then down).
#   2. the absolute build path — fallback if the binary is moved out of
#      the tree. A stale absolute rpath (e.g. from a since-moved repo)
#      is exactly what broke the previous binary; the relative one
#      makes a rebuilt binary location-independent.
LIBS := -L$(SPRAL_BUILD) -lspral \
        -L/opt/homebrew/lib -lmetis \
        -framework Accelerate \
        -lpthread \
        -Wl,-rpath,@loader_path/../../ref/spral/build \
        -Wl,-rpath,$(abspath $(SPRAL_BUILD))

all: spral ssids_bench

ssids_bench: ssids_bench.f90 $(SPRAL_BUILD)/libspral.dylib
	$(FC) $(FFLAGS) -o $@ $< $(LIBS)

# Configure and build SPRAL via meson on first run.
spral: $(SPRAL_BUILD)/libspral.dylib

$(SPRAL_BUILD)/libspral.dylib:
	CC=gcc-15 CXX=g++-15 FC=gfortran meson setup $(SPRAL_BUILD) $(SPRAL_DIR) \
	    -Dgpu=false -Dexamples=false -Dtests=false \
	    -Dlibmetis_path=/opt/homebrew/lib --buildtype=release || true
	meson compile -C $(SPRAL_BUILD)

clean:
	rm -f ssids_bench

spral-clean:
	rm -rf $(SPRAL_BUILD)

clean-all: clean spral-clean

.PHONY: all spral clean spral-clean clean-all
