.PHONY: all
all: cuda

cuda: | cuda.ptx

CPPFLAGS += -isystem/opt/cuda/include/
CFLAGS += -Wall
LDLIBS += -lcuda

NVCC = nvcc
NVCCFLAGS = -arch=sm_35

%.ptx: %.cu
	$(NVCC) $(NVCCFLAGS) -ptx -o $@ $<

.PHONY: clean
clean:
	$(RM) cuda.ptx