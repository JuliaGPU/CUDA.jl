# Conditional use

CUDA.jl is special in that developers may want to depend on the GPU toolchain even though
users might not have a GPU. In this section, we describe two different usage scenarios and
how to implement them. Key to remember is that CUDA.jl **will always load**, which means you
need to manually **check if the package is functional**.

Because CUDA.jl always loads, even if the user doesn't have a GPU or CUDA, you should just
depend on it like any other package (and not use, e.g., Requires.jl). This ensures that
breaking changes to the GPU stack will be taken into account by the package resolver when
installing your package.

If you unconditionally use the functionality from CUDA.jl, you will get a run-time error
in the case the package failed to initialize. For example, on a system without CUDA:

```julia
julia> using CUDA
julia> CUDA.version()
 ┌ Error: Could not initialize CUDA
│   exception =
│    could not load library "libcuda"
│    libcuda.so: cannot open shared object file: No such file or directory
└ @ CUDA CUDA.jl/src/initialization.jl:99
```

To avoid this, you should call `CUDA.functional()` to inspect whether the package is
functional and condition your use of GPU functionality on that. Let's illustrate with two
scenario's, one where having a GPU is required, and one where it's optional.


## Scenario 1: GPU is required

If your application requires a GPU, and its functionality is not designed to work without
CUDA, you should just import the necessary packages and inspect if they are functional:

```julia
using CUDA
@assert CUDA.functional(true)
```

Passing `true` as an argument makes CUDA.jl display why initialization might have failed.

If you are developing a package, you should take care only to perform this check at run
time. This ensures that your module can always be precompiled, even on a system without a
GPU:

```julia
module MyApplication

using CUDA

__init__() = @assert CUDA.functional(true)

end
```

This of course also implies that you should avoid any calls to the GPU stack from global
scope, since the package might not be functional.


## Scenario 2: GPU is optional

If your application does not require a GPU, and can work without the CUDA packages, there is
a tradeoff. As an example, let's define a function that uploads an array to the GPU if
available:

```julia
module MyApplication

using CUDA

if CUDA.functional()
    to_gpu_or_not_to_gpu(x::AbstractArray) = CuArray(x)
else
    to_gpu_or_not_to_gpu(x::AbstractArray) = x
end

end
```

This works, but cannot be simply adapted to a scenario with precompilation on a system
without CUDA. One option is to evaluate code at run time:

```julia
function __init__()
    if CUDA.functional()
        @eval to_gpu_or_not_to_gpu(x::AbstractArray) = CuArray(x)
    else
        @eval to_gpu_or_not_to_gpu(x::AbstractArray) = x
    end
end
```

However, this causes compilation at run-time, and might negate much of the advantages that
precompilation has to offer. Instead, you can use a global flag:

```julia
const use_gpu = Ref(false)
to_gpu_or_not_to_gpu(x::AbstractArray) = use_gpu[] ? CuArray(x) : x

function __init__()
    use_gpu[] = CUDA.functional()
end
```

The disadvantage of this approach is the introduction of a type instability.
