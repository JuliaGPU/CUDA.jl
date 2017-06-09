.PHONY: all
all: cuda

cuda: | cuda.ptx

CFLAGS += -Wall
LDLIBS += -lcuda

NVCC = nvcc
NVCCFLAGS = -arch=sm_35 -ccbin=$(CXX)

%.ptx: %.cu
	$(NVCC) $(NVCCFLAGS) -ptx -o $@ $<

.PHONY: clean
clean:
	$(RM) cuda.ptx