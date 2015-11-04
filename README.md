CUDA.jl
=======

CUDA programming interface for Julia, with native code execution support.

This package wraps key functions in CUDA driver API for Julia, and provides
support for writing native Julia kernels for execution on an NVIDIA GPU. Note
that this requires a version of Julia capable of generating PTX code, such as
[maleadt/julia](https://github.com/maleadt/julia). While this remains a work in
progress, simple use is ready.


## Setup

1. Install the NVIDIA driver, and make sure `libcuda` is in your library loading
   path. If NVIDIA hardware is unavailable, you can use the [GPU Ocelot
   emulator](https://github.com/maleadt/gpuocelot) instead.

2. Install the CUDA toolkit (not all versions are supported, you might need to
   install an older version). This is independent from whether you're using the
   NVIDIA driver or GPU Ocelot emulator. Make sure the location of `nvcc` is
   part of your `PATH` environment variable

   If the toolkit is installed in a non-standard location, you also need to
   define the `NVVMIR_LIBRARY_DIR` environment variable, pointing to the
   directory containing the `libdevice` bitcode files. These files are only part
   of recent CUDA toolkits (version 5.5 or greater), so if you're using an older
   version you will need to get a hold of these files.

2. Check-out this package in Julia:

   ```julia
   Pkg.clone($CHECKOUT_DIRECTORY)
   ```

3. Test if everything works as expected by running `tests/runtests.jl` (you can
   also do this using `Pkg.test`). In both cases, defining the `PERFORMANCE`
   environment variable activates performance measurements.


## Example

The following example shows how to add two vectors on the GPU:

**Writing the kernel**

First you have to write the computation kernel and mark it `@target ptx`:

```julia
using CUDA

@target ptx function kernel_vadd(a::CuDeviceArray{Float32}, b::CuDeviceArray{Float32},
                                 c::CuDeviceArray{Float32})
    i = blockIdx().x +  (threadIdx().x-1) * gridDim().x
    c[i] = a[i] + b[i]

    return nothing
end

```

**Launching the kernel**

Using the `@cuda` macro, you can launch the kernel on a GPU of your choice:

```julia
using CUDA

# select a CUDA device
dev = CuDevice(0)

# create contexts on the selected device
ctx = CuContext(dev)
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

# finalize: destroy contexts
destroy(cgctx)
destroy(ctx)
```

See `tests/native.jl` for more comprehensive examples.


## Advanced use

### Override autodetection

The library tries to autodetect some properties of the available toolkit and
hardware when it is initialized. If these results are wrong, or you want to
override settings, there are several environment variables to use:

* `CUDA_FORCE_API_VERSION`: this mimics the macro from `cuda.h`, with the same
  format.
* `CUDA_FORCE_GPU_TRIPLE` and `CUDA_FORCE_GPU_ARCH`: GPU triple and architecture
  to pass to LLVM, defaults to respectively hard-coded `nvptx64-nvidia-cuda` and
  auto-detected value based on the actual device capabilities.
