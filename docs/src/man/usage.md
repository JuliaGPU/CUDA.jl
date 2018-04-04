# Usage


## Quick start

First you have to write the kernel function and make sure it only uses features from the
CUDA-supported subset of Julia:

```julia
using CUDAnative

function kernel_vadd(a, b, c)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    c[i] = a[i] + b[i]

    return nothing
end

```

Using the `@cuda` macro, you can launch the kernel on a GPU of your choice:

```julia
using CUDAdrv, CUDAnative
using Base.Test

# CUDAdrv functionality: generate and upload data
a = round.(rand(Float32, (3, 4)) * 100)
b = round.(rand(Float32, (3, 4)) * 100)
d_a = CuArray(a)
d_b = CuArray(b)
d_c = similar(d_a)  # output array

# run the kernel and fetch results
# syntax: @cuda [kwargs...] kernel(args...)
@cuda threads=12 kernel_vadd(d_a, d_b, d_c)

# CUDAdrv functionality: download data
# this synchronizes the device
c = Array(d_c)

@test a+b ≈ c
```

This code is executed in a default, global context for the first device in your
system. Similar to `cudaSetDevice`, you can switch devices by calling
CUDAnative's `device!` function:

```julia
# change the active device
device!(1)

# the same, but only temporarily
device!(2) do
    # ...
end
```

Contrary to CUDA however, a context for the first device is always initialized
when loading the package. If you want to avoid this, launch Julia with the
environment variable `CUDANATIVE_INITIALIZE` set to `false`.



## Julia support

Only a limited subset of Julia is supported by this package. This subset is undocumented, as
it is too much in flux.

In general, GPU support of Julia code is determined by the language features used by the
code. Several parts of the language are downright disallowed, such as calls to the Julia
runtime, or garbage allocations. Other features might get reduced in strength, eg. throwing
exceptions will result in a `trap`.

If your code is incompatible with GPU execution, the compiler will mention the unsupported
feature, and where the use came from:

```
julia> foo(i) = (print("can't do this"); return nothing)
foo (generic function with 1 method)

julia> @cuda foo(1)
ERROR: error compiling foo: error compiling print: generic call to unsafe_write requires the runtime language feature
```

In addition, the JIT doesn't support certain modes of compilation. For example, recursive
functions require a proper cached compilation, which is currently absent.


## CUDA support

Not all of CUDA is supported, and because of time constraints the supported subset is again
undocumented. The following (incomplete) list details the support and their CUDAnative.jl
names. Most are implemented in `intrinsics.jl`, so have a look at that file for a more up to
date list:

* Indexing: `threadIdx().{x,y,z}`, `blockDim()`, `blockIdx()`, `gridDim()`, `warpsize()`
* Shared memory: `@cuStaticSharedMemory`, `@cuDynamicSharedMemory`
* Array type: `CuDeviceArray` (converted from input `CuArray`s, or shared memory)
* I/O: `@cuprintf`
* Synchronization: `sync_threads`
* Communication: `vote_{all,any,ballot}`
* Data movement: `shfl_{up,down,bfly,idx}`

### `libdevice`

In addition to the native intrinsics listed above, math functionality from `libdevice` is
wrapped and part of CUDAnative. For now, you need to fully qualify function calls to these
intrinsics, which provide similar functionality to some of the low-level math functionality
of Base which would otherwise call out to `libm`.
