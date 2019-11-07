# Conditional Usage

The GPU stack and its packages are special in that developers may want to depend on them
even though users might not have a GPU. In this section, we describe three different usage
scenarios and how to implement them. Key to remember is that the CUDA packages **will always
load**, which means you need to manually **check if they are functional**.

Because the packages are always loadable, you should just depend on them like any other
package (and not use, e.g., Requires.jl). This ensures that breaking changes to the GPU
stack will be taken into account by the package resolver when installing your package.

If you want to know *why* one of the GPU packages fails to load, enable debug logging:

```
$ julia -e "using CUDAdrv; @show CUDAdrv.functional()"

$ JULIA_DEBUG=all julia -e "using CUDAdrv; @show CUDAdrv.functional()"
┌ Debug: CUDAdrv.jl failed to initialize; the package will not be functional.
│   exception =
│    error compiling __hidden_init__: error compiling cuInit: could not load library "libcuda"
│    libcuda.so: cannot open shared object file: No such file or directory
└ @ CUDAdrv ~/Julia/pkg/CUDAdrv/src/CUDAdrv.jl:51
CUDAdrv.functional() = false
```


## Required

If your application requires a GPU, and its functionality is not designed to work without
CUDA, you should just import the necessary packages and inspect if they are functional:

```julia
using CuArrays
@assert CuArrays.functional()
```

If your application may need to be precompiled on a system without CUDA (e.g. the log-in
node of a cluster, or the build phase of a container), you should only check at run time
whether the packages work as expected:

```julia
module MyApplication

using CuArrays

__init__() = @assert CuArrays.functional()

end
```

This of course also implies that you should avoid any calls to the GPU stack from global
scope, since the packages might not be functional.


## Optional

IF your application does not require a GPU, and can work without the CUDA packages, there is
a tradeoff. As an example, let's define a function that uploads an array to the GPU if
available:

```julia
module MyApplication

using CuArrays

if CuArrays.functional()
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
    if CuArrays.functional()
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
    use_gpu[] = CuArrays.functional()
end
```

The disadvantage of this approach is the introduction of a type instability.
