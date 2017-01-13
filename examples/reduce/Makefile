.phony: all
all: reduce

NVCC=$(shell which nvcc)
NVCCFLAGS += -arch=sm_20 -O3

%: %.cu
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) $(NVCC_LDLIBS) -o $@ $^
