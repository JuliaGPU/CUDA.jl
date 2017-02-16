SRCS=$(wildcard *.cu)
OBJS=$(SRCS:.cu=.ptx)

CUDA_ROOT=/usr
NVCC=$(CUDA_ROOT)/bin/nvcc
NVCCFLAGS=-arch=sm_20


.PHONY: all
all: $(OBJS)

.PHONY: clean
clean:
	$(RM) $(OBJS)


%.ptx: %.cu
	$(NVCC) $(NVCCFLAGS) -ptx $^ -o $@
	sed -i -e '/\.file/d' -e '/\.loc/d' -e '/^\/\//d' $@
