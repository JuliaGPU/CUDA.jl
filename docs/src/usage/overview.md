# [Overview](@id UsageOverview)

The CUDA.jl package provides three distinct, but related, interfaces for CUDA programming:

- the `CuArray` type: for programming with arrays;
- native kernel programming capabilities: for writing CUDA kernels in Julia;
- CUDA API wrappers: for low-level interactions with the CUDA libraries.

Much of the Julia CUDA programming stack can be used by just relying on the `CuArray` type,
and using platform-agnostic programming patterns like `broadcast` and other array
abstractions. Only once you hit a performance bottleneck, or some missing functionality, you
might need to write a custom kernel or use the underlying CUDA APIs.


## The `CuArray` type

The `CuArray` type is an essential part of the toolchain. Primarily, it is used to manage
GPU memory, and copy data from and back to the CPU:

```julia
a = CuArray{Int}(undef, 1024)

# essential memory operations, like copying, filling, reshaping, ...
b = copy(a)
fill!(b, 0)
@test b == CUDA.zeros(Int, 1024)

# automatic memory management
a = nothing
```

Beyond memory management, there are a whole range of array operations to process your data.
This includes several higher-order operations that take other code as arguments, such as
`map`, `reduce` or `broadcast`. With these, it is possible to perform kernel-like operations
without actually writing your own GPU kernels:

```julia
a = CUDA.zeros(1024)
b = CUDA.ones(1024)
a.^2 .+ sin.(b)
```

When possible, these operations integrate with existing vendor libraries such as CUBLAS and
CURAND. For example, multiplying matrices or generating random numbers will automatically
dispatch to these high-quality libraries, if types are supported, and fall back to generic
implementations otherwise.


## Kernel programming with `@cuda`

If an operation cannot be expressed with existing functionality for `CuArray`, or you need
to squeeze every last drop of performance out of your GPU, you can always write a custom
kernel. Kernels are functions that are executed in a massively parallel fashion, and are
launched by using the `@cuda` macro:

```julia
a = CUDA.zeros(1024)

function kernel(a)
    i = threadIdx().x
    a[i] += 1
    return
end

@cuda threads=length(a) kernel(a)
```

These kernels give you all the flexibility and performance a GPU has to offer, within a
familiar language. However, not all of Julia is supported: you (generally) cannot allocate
memory, I/O is disallowed, and badly-typed code will not compile. As a general rule of
thumb, keep kernels simple, and only incrementally port code while continuously verifying
that it still compiles and executes as expected.


## CUDA API wrappers

For advanced use of the CUDA, you can use the driver API wrappers in CUDA.jl. Common
operations include synchronizing the GPU, inspecting its properties, starting the profiler,
etc. These operations are low-level, but for your convenience wrapped using high-level
constructs. For example:

```julia
CUDA.@profile begin
    # code that runs under the profiler
end

# or

for device in CUDA.devices()
    @show capability(device)
end
```

If such high-level wrappers are missing, you can always access the underling C API
(functions and structures prefixed with `cu`) without having to ever exit Julia:

```julia
version = Ref{Cint}()
CUDA.cuDriverGetVersion(version)
@show version[]
```
