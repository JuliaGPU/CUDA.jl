# [Overview](@id UsageOverview)

There are three key packages that make up the Julia stack for CUDA programming:

* [CUDAdrv.jl](https://github.com/JuliaGPU/CUDAdrv.jl) for interfacing with the CUDA APIs
* [CUDAnative.jl](https://github.com/JuliaGPU/CUDAnative.jl) for writing CUDA kernels
* [CuArrays.jl](https://github.com/JuliaGPU/CuArrays.jl) for working with CUDA arrays

You probably won't need all three of these packages: Much of the Julia CUDA programming
stack can be used by just relying on the `CuArray` type, and using platform-agnostic
programming patterns like `broadcast` and other array abstractions.


## CuArrays.jl

The CuArrays.jl package provides an essential part of the toolchain: an array type for
managing data on the GPU and performing operations on its elements. Every application should
use this type, if only to manage memory because it is much easier then doing manual memory
management:

```julia
using Cuarrays

a = CuArray{Int}(undef, 1024)

# essential memory operations, like copying, filling, reshaping, ...
b = copy(a)
fill!(b, 0)
@test b == CuArrays.zeros(Int, 1024)

# automatic memory management
a = nothing
```

Beyond memory management, CuArrays also implements a whole range of array operations. This
includes several higher-order operations that take other code as arguments, such as `map`,
`reduce` or `broadcast`. With these, it is possible to perform kernel-like operations
without actually writing your own GPU kernels:

```julia
a = CuArrays.zeros(1024)
b = CuArrays.ones(1024)
a.^2 .+ sin.(b)
```

Finally, CuArrays.jl also integrates with existing vendor libraries such as CUBLAS and
CURAND. Common operations such as matrix multiplication or generating random numbers will
automatically dispatch to these high-quality libraries where possible, and fall back to
generic implementations if necessary.


## CUDAnative.jl

If an operation cannot be expressed with existing operations in CuArrays.jl, or you need to
squeeze every last drop of performance out of your GPU, you can use CUDAnative.jl to write a
custom kernel. Kernels are functions that are executed in a massively parallel fashion, and
are launched by using the `@cuda` macro:

```julia
using CuArrays, CUDAnative

a = CuArrays.zeros(1024)

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


## CUDAdrv.jl

For advanced use of the CUDA APIs, you can use the CUDAdrv.jl package (itself used by both
CuArrays.jl and CUDAnative.jl). Common operations include synchronizing the GPU, inspecting
its properties, starting the profiler, etc. These operations are low-level, but for your
convenience wrapped using high-level constructs. For example:

```julia
using CUDAdrv

CUDAdrv.@profile begin
    # code that runs under the profiler
end

# or

for device in CUDAdrv.devices()
    @show capability(device)
end
```

If such high-level wrappers are missing, you can always access the underling C API
(functions and structures prefixed with `cu`) without having to ever exit Julia:

```julia
version = Ref{Cint}()
CUDAdrv.cuDriverGetVersion(version)
@show version[]
```


## Others

Several other packages exist serving more specific purposes, and are not worth discussing in
detail here:

- [CUDAapi.jl](https://github.com/JuliaGPU/CUDAapi.jl) for reusable CUDA API programming constructs
- [NCCL.jl](https://github.com/JuliaGPU/NCCL.jl) wrapping the NCCL library with collective communication primitives
