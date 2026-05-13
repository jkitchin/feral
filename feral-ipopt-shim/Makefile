# Build Ipopt 3.14 (vendored at ../ref/Ipopt) with feral as the
# linear_solver=feral option.
#
# Targets:
#   make all          — build feral staticlib, patch Ipopt, configure,
#                       and build Ipopt.
#   make hs071-feral  — run the hs071 sample NLP with linear_solver=feral.
#   make clean        — remove the Ipopt build dir + revert the patch.
#
# Variables (override on command line):
#   IPOPT_SRC   = ../ref/Ipopt
#   IPOPT_BUILD = $(IPOPT_SRC)/build-feral
#   FERAL_ROOT  = ..  (the feral crate root)

SHIM_DIR    := $(CURDIR)
IPOPT_SRC   ?= $(SHIM_DIR)/../ref/Ipopt
IPOPT_BUILD ?= $(IPOPT_SRC)/build-feral
FERAL_ROOT  ?= $(SHIM_DIR)/..
FERAL_LIB   ?= $(FERAL_ROOT)/target/release/libferal.a

LINSOLV_DIR := $(IPOPT_SRC)/src/Algorithm/LinearSolvers
SHIM_HPP    := $(LINSOLV_DIR)/IpFeralSolverInterface.hpp
SHIM_CPP    := $(LINSOLV_DIR)/IpFeralSolverInterface.cpp
SHIM_CHDR   := $(LINSOLV_DIR)/feral_capi.h

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
NATIVE_LIBS := -lSystem -lc -lm
else
NATIVE_LIBS := -ldl -lpthread -lm
endif

.PHONY: all feral-lib install-shim configure-ipopt build-ipopt hs071-feral clean

all: build-ipopt

# Step a — build feral as a staticlib.
feral-lib $(FERAL_LIB):
	cd $(FERAL_ROOT) && cargo build --release

# Step b — install shim source into Ipopt source tree and apply patch.
$(IPOPT_SRC)/.feral-patched: $(FERAL_LIB) patches/ipopt-feral.patch \
                             include/IpFeralSolverInterface.hpp \
                             src/IpFeralSolverInterface.cpp \
                             include/feral_capi.h
	# Rename header to Ipopt's "Ip<Name>.hpp" convention so the
	# include line in IpAlgBuilder.cpp resolves correctly.
	cp include/IpFeralSolverInterface.hpp $(SHIM_HPP)
	cp src/IpFeralSolverInterface.cpp     $(SHIM_CPP)
	cp include/feral_capi.h               $(SHIM_CHDR)
	cd $(IPOPT_SRC) && patch -p1 --forward < $(SHIM_DIR)/patches/ipopt-feral.patch \
	    || { echo "patch failed or already applied; check $(IPOPT_SRC)"; exit 1; }
	# Patch directly modifies Makefile.in too, so no autoreconf needed.
	touch $@

# Step c — configure Ipopt, point at libferal.a.
$(IPOPT_BUILD)/config.status: $(IPOPT_SRC)/.feral-patched
	mkdir -p $(IPOPT_BUILD)
	# We don't pass FERAL_LIB here — Ipopt's libtool would bundle
	# libferal.a as a nested archive member inside libipopt.a,
	# which downstream linkers can't parse. We link feral in at
	# the final executable step instead (see hs071-feral target).
	cd $(IPOPT_BUILD) && $(IPOPT_SRC)/configure \
	    --disable-shared --enable-static \
	    --without-hsl --without-spral --without-pardiso \
	    --without-asl \
	    ADD_CXXFLAGS="-I$(SHIM_DIR)/include"

# Step d — build Ipopt.
build-ipopt: $(IPOPT_BUILD)/config.status
	$(MAKE) -C $(IPOPT_BUILD) -j

# Step e — run the bundled hs071 sample with linear_solver=feral.
# The hs071_cpp example builds itself under $(IPOPT_BUILD)/examples;
# we set the linear_solver via an ipopt.opt file dropped in the cwd.
HS071_DIR := $(IPOPT_BUILD)/examples/hs071_cpp
# The shipped example Makefile hardcodes the system-installed
# /usr/local/lib/pkgconfig path — it would link against a stale
# installed Ipopt, not our build. Compile and link directly against
# our build tree instead.
IPOPT_LIB := $(IPOPT_BUILD)/src/.libs/libipopt.a
IPOPT_INC := -I$(IPOPT_SRC)/src/Interfaces \
             -I$(IPOPT_SRC)/src/Common \
             -I$(IPOPT_SRC)/src/LinAlg \
             -I$(IPOPT_BUILD)/src
ifeq ($(UNAME_S),Darwin)
HS071_LDFLAGS := -framework Accelerate
else
HS071_LDFLAGS := -llapack -lblas -lgfortran
endif

hs071-feral: build-ipopt
	g++ -O2 -DNDEBUG $(IPOPT_INC) \
	    -c -o $(HS071_DIR)/hs071_main.o $(HS071_DIR)/hs071_main.cpp
	g++ -O2 -DNDEBUG $(IPOPT_INC) \
	    -c -o $(HS071_DIR)/hs071_nlp.o  $(HS071_DIR)/hs071_nlp.cpp
	g++ -O2 -DNDEBUG \
	    -o $(HS071_DIR)/hs071_cpp \
	    $(HS071_DIR)/hs071_main.o $(HS071_DIR)/hs071_nlp.o \
	    $(IPOPT_LIB) $(FERAL_LIB) $(HS071_LDFLAGS) $(NATIVE_LIBS)
	@echo "linear_solver feral" > $(HS071_DIR)/ipopt.opt
	cd $(HS071_DIR) && ./hs071_cpp

clean:
	rm -rf $(IPOPT_BUILD)
	cd $(IPOPT_SRC) && patch -R -p1 --forward < $(SHIM_DIR)/patches/ipopt-feral.patch \
	    || echo "(no patch to revert)"
	rm -f $(IPOPT_SRC)/.feral-patched
	rm -f $(SHIM_HPP) $(SHIM_CPP) $(SHIM_CHDR)
