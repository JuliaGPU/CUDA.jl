# CUDAnative.jl

Code Coverage: [![Coverage Status](https://codecov.io/gh/JuliaGPU/CUDAnative.jl/coverage.svg)](https://codecov.io/gh/JuliaGPU/CUDAnative.jl)

This package provides support for compiling and executing native Julia kernels on CUDA
hardware. It is a work in progress, highly experimental, and for now requires a version of
Julia with the necessary compiler functionality (ie. the development tree from
[PR #18338](https://github.com/JuliaLang/julia/pull/18338)).


## Installation

1. Install the NVIDIA driver, and make sure `libcuda` is in your library loading path.

2. Install the CUDA toolkit, making sure it contains the device library bitcode files, named
   `libdevice.*.bc`. If the toolkit is installed in a nonstandard location, you will need to
   define the `NVVMIR_LIBRARY_DIR` environment variable, pointing to the directory
   containing these bitcode files.

  Note that these files are only part of recent CUDA toolkits (version 5.5 or greater). If
  you are using an older version, you will need to copy over those files from another
  system.

3. Compile a version of Julia with external language support, and use that `julia` binary
   for all future steps.

4. Clone and test this package in Julia:

   ```julia
   Pkg.clone("https://github.com/JuliaGPU/CUDAnative.jl.git")
   Pkg.test("CUDAnative")
   ```

   NOTE: as there is no released version of CUDAnative yet, you will need to check-out the
   latest versions of some dependencies:

   ```julia
   Pkg.checkout("CUDAdrv")
   Pkg.checkout("LLVM")
   ```


## Quick start guide

The following example shows how to add two vectors on the GPU.

**Writing the kernel**

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

**Launching the kernel**

Using the `@cuda` macro, you can launch the kernel on a GPU of your choice:

```julia
using CUDAdrv, CUDAnative

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
# syntax: @cuda device (dims...) kernel(args...)
@cuda dev (1,12) kernel_vadd(d_a, d_b, d_c)

# CUDAdrv functionality: download data
# this synchronizes the device
c = Array(d_c)

@test a+b ≈ c

free(...)
destroy(ctx)
```

See `examples` or `tests/native.jl` for more comprehensive examples.


## Usage

This section documents some specific details on how to use the CUDAnative.jl package, and
what to keep in mind.

Note that this library is not meant to export a high-level interface for using GPUs
transparently, instead it serves to write high-performance GPU kernels in Julia and manage
their execution. Consequently, *you need to understand how GPUs work*, and more specifically
*you need to know your way around CUDA*. Even though many components are made easier to work
with, it sits at an abstraction level similar to CUDA's.

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
julia> foo(i) = "foo$i"
foo (generic function with 1 method)

julia> CUDAnative.code_llvm(foo, (Int,))
ERROR: error compiling foo: error compiling #print_to_string#312: emit_allocobj for strings/io.jl:92 requires the dynamic_alloc language feature, which is disabled
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


## Debugging

### `trap` and kernel launch failures

Exceptions, like the ones being thrown from out-of-bounds accesses, currently just generate
a `trap` instruction which halts the GPU. This might show up as a kernel launch failure, or
an unrelated error in another API call.

If the error is thrown from an array access, and an out-of-bounds access is suspected, it is
useful to turn of bounds checking (`julia --check-bounds=no`) and run the Julia process
under `cuda-memcheck` while enabling debug mode 1 or higher (`julia -g1`). This way,
`cuda-memcheck` will be able to accurately pinpoint the out-of-bounds access, while
specifying the exact location of the access within the active grid and block.

### Output modes

When debugging a failing kernel, or if you want to understand what code is being processed,
you can use CUDAnative's `DEBUG` and `TRACE` modes. These modes are enabled by defining
environment variables with the same name, but do note that their value is cached by
precompilation so you probably need to invoke Julia with `--compilecache=no`:

```
DEBUG=1 julia --compilecache=no

[...]

julia> @cuda dev (1,1) foo(a)
DEBUG: Compiling foo(CUDAnative.CuDeviceArray{Int32,1})
DEBUG: <unknown>:0:0: marked this call a tail call candidate
DEBUG: JIT info log: ...
```

The `TRACE` mode generates even more output, including typed Julia code, LLVM IR and PTX
assembly. These sources can be useful when debugging code generation issues, or pinpointing
use of unsupported language features.

### `code_*` alternatives

CUDAnative provides alternatives to Base's `code_llvm` and `code_native` to inspect
generated GPU code:

```julia
julia> foo(i) = 2*i
foo (generic function with 1 method)

julia> CUDAnative.code_llvm(foo, (Int,))

; Function Attrs: nounwind
define i64 @julia_foo_62262(i64) local_unnamed_addr #0 !dbg !4 {
...
}

julia> CUDAnative.code_native(foo, (Int,))
//
// Generated by LLVM NVPTX Back-End
//

...
```

### Debug info and line-number information

LLVM's NVPTX back-end does not support the undocumented PTX debug format, so we cannot
generate the necessary DWARF sections. This means that debugging generated code with e.g.
`cuda-gdb` will be an unpleasant experience. Nonetheless, the PTX JIT is configured to emit
debug info (which corresponds with `nvcc -g`) when the Julia debug info level is 2 or
higher.

We do however support emitting line number information, which is useful for other CUDA tools
like `cuda-memcheck`. The functionality (which corresponds with `nvcc -lineinfo`) is enabled
when the Julia debug info level is 1 or higher.


## Performance gotcha's

Apart from the standard GPU optimization tips, there are a few special considerations when
using CUDAnative.jl.

### Object arguments

When passing a rich object like a `CuArray` to a GPU kernel, there's a memory allocation and
copy happening behind the scenes. This means that every kernel call is synchronizing, which
can easily kill performance in the case of fine-grained kernels.

Although this issue will probably get fixed in the future, a workaround for now is to ensure
all arguments are `bitstype` (ie. declared as primitive `bitstype` types, not to be confused
with the `isbits` property). Specific to arrays, you can access and pass the underlying
device pointer by means of the `ptr` field of `CuArray` objects, in addition to the size of
the array.
