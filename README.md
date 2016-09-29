# CUDAnative.jl

Code Coverage: [![Coverage Status](https://codecov.io/gh/JuliaGPU/CUDAnative.jl/coverage.svg)](https://codecov.io/gh/JuliaGPU/CUDAnative.jl)

This package provides support for compiling and executing native Julia kernels on CUDA
hardware. It is a work in progress, highly experimental, and for now requires a version of
Julia capable of generating PTX code (ie. the fork at
[JuliaGPU/julia](https://github.com/JuliaGPU/julia)).


## Installation

1. Install the NVIDIA driver, and make sure `libcuda` is in your library loading
   path.

2. Install the CUDA toolkit (not all versions are supported, you might need to install an
   older version).

   You also need to make sure your toolkit ships device library bitcode files, named
  `libdevice.*.bc`. If the toolkit is installed in a nonstandard location, you will need to
  define the `NVVMIR_LIBRARY_DIR` environment variable, pointing to the directory containing
  the `libdevice` bitcode files.

  Note that these files are only part of recent CUDA toolkits (version 5.5 or
  greater), so if you're using an older version you will need to get a hold of
  these files.

3. Install a version of Julia with PTX support, and use that `julia` binary for
   all future steps.

4. Clone and test this package in Julia:

   ```julia
   Pkg.clone("https://github.com/JuliaGPU/CUDAnative.jl.git")
   Pkg.test("CUDAnative")
   ```

   NOTE: if checking out an unreleased version of CUDAnative, you might also need to
   check-out the latest versions of any dependencies:

   ```julia
   Pkg.checkout("CUDAdrv")
   Pkg.checkout("LLVM")
   ```


## Usage

The following example shows how to add two vectors on the GPU:

**Writing the kernel**

First you have to write the computation kernel and mark it `@target ptx`:

```julia
using CUDAnative

@target ptx function kernel_vadd(a, b, c)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    c[i] = a[i] + b[i]

    return nothing
end

```

**Launching the kernel**

Using the `@cuda` macro, you can launch the kernel on a GPU of your choice:

```julia
using CUDAnative

# select a CUDA device
dev = CuDevice(0)

# create context on the selected device
ctx = CuContext(dev)

# generate random arrays and load them to GPU
a = round(rand(Float32, (3, 4)) * 100)
b = round(rand(Float32, (3, 4)) * 100)
d_a = CuArray(a)
d_b = CuArray(b)

# create an array on GPU to store results
d_c = CuArray(Float32, (3, 4))

# run the kernel and fetch results
# syntax: @cuda device (dims...) kernel(args...)
@cuda dev (1,12) kernel_vadd(d_a, d_b, d_c)
c = Array(d_c)

# print the results
println("Results:")
println("a = \n$a")
println("b = \n$b")
println("c = \n$c")

# finalize: destroy context
destroy(ctx)
```

See `examples` or `tests/native.jl` for more comprehensive examples.


## Debugging

### Verbosity

CUDAnative supports a debug and trace mode, where extra checks will be enabled and a lot of
extra output will be generated. Enable these modes by respectively adding `DEBUG=1` or
`TRACE=1` to your environment. Note that these flags are evaluated statically, so you'll
need to start julia with `--compilecache=no` for them to have any effect.

### Debugging symbols and line-number information

Line-number information (cfr. `nvcc -lineinfo`) is only generated when Julia is started with
a debug-level >= 2. Debugging symbols are not implemented, due to the LLVM PTX back-end not
having support for the undocumented PTX DWARF section. This means that CUDA tools should be
able to report location information, but when debugging with `cuda-gdb` you will only have
an assembly view.
