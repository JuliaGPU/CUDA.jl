CUDAnative.jl
=============

**[Build status](https://ci.maleadt.net/buildbot/julia/waterfall?tag=CUDAnative)** (Linux x86-64): [![](https://ci.maleadt.net/buildbot/julia/png?builder=CUDAnative.jl:%20Julia%20master%20(x86-64))](https://ci.maleadt.net/buildbot/julia/builders/CUDAnative.jl%3A%20Julia%20master%20%28x86-64%29)

**Code coverage**: [![Coverage Status](https://codecov.io/gh/JuliaGPU/CUDAnative.jl/coverage.svg)](https://codecov.io/gh/JuliaGPU/CUDAnative.jl)

This package provides support for compiling and executing native Julia kernels on CUDA
hardware. It is a work in progress, and only works on very recent versions of Julia .



Installation
------------

Requirements:

* Julia 0.6 with LLVM 3.9 **built from source**, executed **in tree** (for LLVM.jl)
* NVIDIA driver, providing `libcuda.so` (for CUDAdrv.jl)
* CUDA toolkit

Although that first requirement might sound complicated, it basically means you need to
fetch and compile a copy of Julia 0.6 (refer to [the main repository's
README](https://github.com/JuliaLang/julia/blob/master/README.md#source-download-and-compilation),
checking out the latest tag for 0.6), and execute the resulting `julia` binary in-place
without doing a `make install`. Afterwards, you can do:

```
Pkg.add("CUDAnative")
Pkg.test("CUDAnative")
```

For now, only Linux and macOS are supported.



Quick start
-----------

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

# CUDAdrv functionality: select device, create context
dev = CuDevice(0)
ctx = CuContext(dev)

# CUDAdrv functionality: generate and upload data
a = round.(rand(Float32, (3, 4)) * 100)
b = round.(rand(Float32, (3, 4)) * 100)
d_a = CuArray(a)
d_b = CuArray(b)
d_c = similar(d_a)  # output array

# run the kernel and fetch results
# syntax: @cuda (dims...) kernel(args...)
@cuda (1,12) kernel_vadd(d_a, d_b, d_c)

# CUDAdrv functionality: download data
# this synchronizes the device
c = Array(d_c)

@test a+b ≈ c

destroy(ctx)
```

See the [examples](examples/) folder for more comprehensive examples.



Usage
-----

This section documents some specific details on how to use the CUDAnative.jl package, and
what to keep in mind.

Note that this library is not meant to export a high-level interface for using GPUs
transparently, instead it serves to write high-performance GPU kernels in Julia and manage
their execution. Consequently, **you need to understand how GPUs work**, and more
specifically **you need to know your way around CUDA**. Even though many components are made
easier to work with, it sits at an abstraction level similar to CUDA's.


### Julia support

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

julia> @cuda (1,1) foo(1)
ERROR: error compiling foo: error compiling print: generic call to unsafe_write requires the runtime language feature
```


### CUDA support

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

#### `libdevice`

In addition to the native intrinsics listed above, math functionality from `libdevice` is
wrapped and part of CUDAnative. For now, you need to fully qualify function calls to these
intrinsics, which provide similar functionality to some of the low-level math functionality
of Base which would otherwise call out to `libm`.



Troubleshooting
---------------

You can enable verbose logging using two environment variables:

* `DEBUG`: if set, enable additional (possibly costly) run-time checks, and some more
  verbose output
* `TRACE`: if set, the `DEBUG` level will be activated, in addition with a trace of every
  call to the underlying library

In order to avoid run-time cost for checking the log level, these flags are implemented by
means of global constants. As a result, you **need to run Julia with precompilation
disabled** if you want to modify these flags:

```
$ TRACE=1 julia --compilecache=no examples/vadd.jl
TRACE: CUDAnative.jl is running in trace mode, this will generate a lot of additional output
...
```

Enabling colors with `--color=yes` is also recommended as it color-codes the output.


### `trap` and kernel launch failures

Exceptions, like the ones being thrown from out-of-bounds accesses, currently just generate
a `trap` instruction which halts the GPU. This might show up as a kernel launch failure, or
an unrelated error in another API call.

If the error is thrown from an array access, and an out-of-bounds access is suspected, it is
useful to turn of bounds checking (`julia --check-bounds=no`) and run the Julia process
under `cuda-memcheck` while enabling debug mode 1 (the default value) or higher. This way,
`cuda-memcheck` will be able to accurately pinpoint the out-of-bounds access, while
specifying the exact location of the access within the active grid and block.


### `code_*` alternatives

CUDAnative provides alternatives to Base's `code_llvm` and `code_native` to inspect
generated GPU code:

```julia
julia> foo(a, i) = (a[1] = i; return nothing)
foo (generic function with 1 method)

julia> a = CuArray{Int}(1)

julia> CUDAnative.@code_llvm foo(a, 1)

; Function Attrs: nounwind
define i64 @julia_foo_62405(%CuDeviceArray.2* nocapture readonly, i64) {
...
}

julia> @code_ptx foo(a, 1)
.visible .entry julia_foo_62419(
        .param .u64 julia_foo_62419_param_0,
        .param .u64 julia_foo_62419_param_1
)
{
...
}

julia> @code_sass foo(a, 1)
        code for sm_20
                Function : julia_foo_62539
...
```

Non-macro versions of these reflection entry-points are available as well (ie. `code_llvm`,
etc), but as there's type conversions happening behind the scenes you will need to take care
and perform those conversions manually:

```julia
julia> CUDAnative.code_llvm(foo, (CuArray{Int,1},Int))
ERROR: error compiling foo: ...

julia> CUDAnative.code_llvm(foo, (CuDeviceArray{Int,1},Int))

; Function Attrs: nounwind
define i64 @julia_foo_62405(%CuDeviceArray.2* nocapture readonly, i64) {
...
}
```


### Debug info and line-number information

LLVM's NVPTX back-end does not support the undocumented PTX debug format, so we cannot
generate the necessary DWARF sections. This means that debugging generated code with e.g.
`cuda-gdb` will be an unpleasant experience. Nonetheless, the PTX JIT is configured to emit
debug info (which corresponds with `nvcc -g`) when the Julia debug info level is 2 or
higher (`julia -g2`).

We do however support emitting line number information, which is useful for other CUDA tools
like `cuda-memcheck`. The functionality (which corresponds with `nvcc -lineinfo`) is enabled
when the Julia debug info level is 1 (the default value) or higher.



Bugs and quirks
---------------

### Recursive functions

Recursive functions, either directly or indirectly, are currently not supported.


### Object arguments

When passing a rich object like a `CuArray` to a GPU kernel, there's a memory allocation and
copy happening behind the scenes. This means that every kernel call is synchronizing, which
can easily kill performance in the case of fine-grained kernels.

Although this issue will probably get fixed in the future, a workaround for now is to ensure
all arguments are `bitstype` (ie. declared as primitive `bitstype` types, not to be confused
with the `isbits` property). Specific to arrays, you can access and pass the underlying
device pointer by means of the `ptr` field of `CuArray` objects, in addition to the size of
the array:

```julia
function inc_slow(a)
    a[threadIdx().x] += 1

    return nothing
end

@cuda (1,3) inc_slow(d_a)                       # implicit alloc & memcpy


function inc_fast(a_ptr, a_len)
    a = CuDeviceArray(a_len, a_ptr)
    a[threadIdx().x] += 1

    return nothing
end

@cuda (1,3) inc_fast(pointer(d_a), length(d_a)) # no implicit memory ops
```
