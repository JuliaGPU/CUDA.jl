## CUDA.jl

Julia Programming interface for CUDA. 

This package wraps key functions in CUDA Driver API for Julia. While this remains work in progress, simple use is ready.


### Example

The following example shows how one can use this package to add two matrices on GPU.

##### Write CUDA Kernel

First you have to write the computation kernel in CUDA C and save it in a .cu file. Here is a kernel for addition:

```C
// filename: vadd.cu
// a simple CUDA kernel to add two vectors

extern "C"   // ensure function name to be exactly "vadd"
{
	__global__ void vadd(const float *a, const float *b, float *c)
	{
		int i = threadIdx.x + blockIdx.x * blockDim.x;
		c[i] = a[i] + b[i];
	}
} 

```

You may compile the kernel to a PTX file using ``nvcc``, as

```
nvcc -ptx vadd.cu
```

This command would generate a PTX file named ``vadd.ptx``.

##### Run the Kernel in Julia

The following script demonstrates how one can load the kernel and run it in Julia.

```julia
# select a CUDA device
dev = CuDevice(0)

# create a context (like a process in CPU) on the selected device
ctx = create_context(dev)

# load the PTX module (each module can contain multiple kernel functions)
md = CuModule("vadd.ptx")

# retrieve the kernel function "vadd" from the module
vadd = CuFunction(md, "vadd")

# generate random arrays and load them to GPU
a = round(rand(Float32, (3, 4)) * 100)
b = round(rand(Float32, (3, 4)) * 100)
ga = CuArray(a)
gb = CuArray(b)

# create an array on GPU to store results
gc = CuArray(Float32, (3, 4))

# run the kernel vadd
# syntax: launch(kernel, grid_size, block_size, arguments)
# here, grid_size and block_size can be an integer or a tuple of integers
launch(vadd, 12, 1, (ga, gb, gc))

# download the results from GPU
c = to_host(gc)   # c is a Julia array on CPU (host)

# release GPU memory
free(ga)
free(gb)
free(gc)

# print the results
println("Results:")
println("a = \n$a")
println("b = \n$b")
println("c = \n$c")

# finalize: unload module and destroy context
unload(md)
destroy(ctx)
```

This is a relatively low-level API and is designed for people who have some understanding of CUDA programming to write/migrate CUDA codes in Julia. Compared to CUDA C, the interface has been greatly simplified. 

