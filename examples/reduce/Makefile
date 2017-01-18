.phony: all
all: reduce.so

NVCC=$(shell which nvcc)
NVCCFLAGS += -arch=sm_61 -O3

%.so: %.cu
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) -o $@ --compiler-options '-fPIC' -shared $^
