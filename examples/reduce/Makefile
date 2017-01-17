.phony: all
all: reduce reduce.so reduce.ptx

NVCC=$(shell which nvcc)
NVCCFLAGS += -arch=sm_61 -O3

%: %.cu
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) $(NVCC_LDLIBS) -o $@ $^

%.ptx: %.cu
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) -ptx -o $@ $^

%.so: %.cu
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) $(NVCC_LDLIBS) -o $@ --compiler-options '-fPIC' -shared $^