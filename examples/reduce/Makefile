.CLEAN: all
all: reduce.so reduce.cu.ptx

NVCC=$(shell which nvcc)
NVCCFLAGS += -arch=sm_61 -O3

%.so: %.cu
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) -o $@ --compiler-options '-fPIC' -shared $^

%.cu.ptx: %.cu
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) -ptx -o $@ $^

.CLEAN: clean
clean:
	$(RM) reduce.so reduce.cu.ptx
