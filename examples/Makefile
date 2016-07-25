OBJS=vadd.ptx vadd_parent.ptx vadd_child.ptx

NVCC=nvcc
NVCCFLAGS=


.PHONY: all
all: $(OBJS)

.PHONY: clean
clean:
	$(RM) $(OBJS)


%.ptx: %.cu
	$(NVCC) $(NVCCFLAGS) -ptx $^ -o $@