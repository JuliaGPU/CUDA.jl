## CUDA.jl

CUDA programming interface for Julia, with native code execution support.

This package wraps key functions in CUDA driver API for Julia, and provides
support for writing native Julia kernels for execution on a NVIDIA GPU. Note
that this requires a version of Julia capable of generating PTX code, such as
[maleadt/julia](https://github.com/maleadt/julia). While this remains a work in
progress, simple use is ready.

### Setup

1. Install CUDA driver, and make sure `libcuda` is in your library loading
   path. Alternatively, you can use GPU Ocelot (see
   [maleadt/gpuocelot](https://github.com/maleadt/gpuocelot)) instead when you
   don't have CUDA-capable hardware at hand.

2. Checkout this package in Julia:

```julia
Pkg.clone($CHECKOUT_DIRECTORY)
```

3. Test whether it works by running the tests in `tests/runtests.jl` (you can
   also do this using `Pkg.test`). In both cases, defining the `PERFORMANCE`
   environment variable activates performance measurements.

4. Enjoy.

Note that the compiler requires `libdevice` to link kernel functions. This
library is only part of recent CUDA toolkits (version 5.5 or greater). If you
use older an older CUDA release (for example because you use the GPU Ocelot
emulator which only supports up to CUDA 5.0) you _will_ need to get a hold of
these files. Afterwards, you can point Julia to these files using the
NVVMIR_LIBRARY_DIR environment variable.


### Example

The following example shows how one can use this package to add two vectors on
GPU.

##### Writing the kernel

First you have to write the computation kernel and mark it `@target ptx`:

```julia
using CUDA

@target ptx function kernel_vadd(a::CuDeviceArray{Float32}, b::CuDeviceArray{Float32},
                                 c::CuDeviceArray{Float32})
    i = blockId_x() + (threadId_x()-1) * numBlocks_x()
    c[i] = a[i] + b[i]

    return nothing
end
```

##### Launching the kernel

Using the `@cuda` macro, you can launch the kernel on a GPU of your choice:

```julia
using CUDA

# select a CUDA device
dev = CuDevice(0)

# create a context (like a process in CPU) on the selected device
ctx = CuContext(dev)

# initialize native code generation support
cgctx = CuCodegenContext(ctx, dev)

# generate random arrays and load them to GPU
a = round(rand(Float32, (3, 4)) * 100)
b = round(rand(Float32, (3, 4)) * 100)

# create an array on GPU to store results
c = Array(Float32, (3, 4))

# run the kernel vadd
# syntax: @cuda (dims...) kernel(args...)
# the CuIn/CuOut/CuInOut modifiers are optional, but improve performance
@cuda (12, 1) kernel_vadd(CuIn(a), CuIn(b), CuOut(c))

# print the results
println("Results:")
println("a = \n$a")
println("b = \n$b")
println("c = \n$c")

# finalize: unload module and destroy context
destroy(ctx)
```

See `tests/native.jl` for more comprehensive examples.
