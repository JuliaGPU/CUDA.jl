```@meta
EditURL = "https://github.com/JuliaGPU/CUDA.jl/blob/master/docs/src/tutorials/custom_structs.jl"
```

# Using custom structs

This tutorial shows how to use custom structs on the GPU. Our example will be a one dimensional
interpolation. Lets start with the CPU version:

```@example custom_structs
using CUDA

struct Interpolate{A}
    xs::A
    ys::A
end

function (itp::Interpolate)(x)
    i = searchsortedfirst(itp.xs, x)
    i = clamp(i, firstindex(itp.ys), lastindex(itp.ys))
    @inbounds itp.ys[i]
end

xs_cpu = [1.0, 2.0, 3.0]
ys_cpu = [10.0,20.0,30.0]
itp_cpu = Interpolate(xs_cpu, ys_cpu)
pts_cpu = [1.1,2.3]
result_cpu = itp_cpu.(pts_cpu)
```

Ok the CPU code works, let's move our data to the GPU:

```@example custom_structs
itp = Interpolate(CuArray(xs_cpu), CuArray(ys_cpu))
pts = CuArray(pts_cpu);
nothing #hide
```

If we try to call our interpolate `itp.(pts)`, we get an error however:
```
...
KernelError: passing and using non-bitstype argument
...
```
Why does it throw an error? Our calculation involves
a custom type `Interpolate{CuArray{Float64, 1}}`.
At the end of the day all arguments of a CUDA kernel need to
be bitstypes. However we have

```@example custom_structs
isbitstype(typeof(itp))
```

How to fix this? The answer is, that there is a conversion mechanism, which adapts objects into
CUDA compatible bitstypes.
It is based on the [Adapt.jl](https://github.com/JuliaGPU/Adapt.jl) package and basic types like `CuArray` already participate in this mechanism. For custom types,
we just need to add a conversion rule like so:

```@example custom_structs
import Adapt
function Adapt.adapt_structure(to, itp::Interpolate)
    xs = Adapt.adapt_structure(to, itp.xs)
    ys = Adapt.adapt_structure(to, itp.ys)
    Interpolate(xs, ys)
end
```

Now our struct plays nicely with CUDA.jl:

```@example custom_structs
result = itp.(pts)
```

It works, we get the same result as on the CPU.

```@example custom_structs
@assert CuArray(result_cpu) == result
```

Alternatively instead of defining `Adapt.adapt_structure` explictly, we could have done
```julia
Adapt.@adapt_structure Interpolate
```
which expands to the same code that we wrote manually.

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

