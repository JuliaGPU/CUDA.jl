OBJS=vadd.ptx

NVCC=nvcc
NVCCFLAGS=


.PHONY: all
all: $(OBJS)

.PHONY: clean
clean:
	$(RM) $(OBJS)


%.ptx: %.cu
	$(NVCC) $(NVCCFLAGS) -ptx $^ -o $@
