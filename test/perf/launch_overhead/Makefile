.PHONY: all
all: cuda

cuda: | cuda.ptx

CPPFLAGS += -isystem/opt/cuda/include/
CFLAGS += -Wall
LDLIBS += -lcuda

NVCC = nvcc
NVCCFLAGS = -arch=sm_61

%.ptx: %.cu
	$(NVCC) $(NVCCFLAGS) -ptx -o $@ $<
