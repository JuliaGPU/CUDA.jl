# CuArrays

[![][buildbot-julia06-img]][buildbot-julia06-url]
[![codecov][codecov-img]][codecov-url]

[buildbot-julia06-img]: http://ci.maleadt.net/shields/build.php?builder=CuArrays-julia06-x86-64bit&name=julia%200.6
[buildbot-julia06-url]: http://ci.maleadt.net/shields/url.php?builder=CuArrays-julia06-x86-64bit
[codecov-img]: https://codecov.io/gh/JuliaGPU/CuArrays.jl/branch/master/graph/badge.svg
[codecov-url]: https://codecov.io/gh/JuliaGPU/CuArrays.jl

CuArrays provides a fully-functional GPU array, which can give significant speedups over normal arrays without code changes. CuArrays are implemented fully in Julia, making the implementation [elegant and extremely generic](http://mikeinnes.github.io/2017/08/24/cudanative.html).

Note that you need to **build Julia 0.6 from source** and have CUDA available to use this package – please see the [CUDAnative.jl](https://github.com/JuliaGPU/CUDAnative.jl) instructions for more details.

```julia
Pkg.add("CuArrays")
```

## Features

```julia
xs = cu(rand(5, 5))
ys = cu[1, 2, 3]
xs_cpu = collect(xs)
```

Because `CuArray` is an `AbstractArray`, it doesn't have much of a learning curve; just use your favourite array ops as usual. The following are supported (on arbitrary numbers of arguments, dimensions etc):

* Conversions and `copy!` with CPU arrays
* General indexing (`xs[1:2, 5, :]`)
* `permutedims`
* Concatenation (`vcat(x, y)`, `cat(3, xs, ys, zs)`)
* `map`, fused broadcast (`zs .= xs.^2 .+ ys .* 2`)
* `fill!(xs, 0)`
* Reduction over dimensions (`reducedim(+, xs, 3)`, `sum(x -> x^2, xs, 1)` etc)
* Reduction to scalar (`reduce(*, 1, xs)`, `sum(xs)` etc)
* Various BLAS operations (matrix\*matrix, matrix\*vector)
* FFTs, using the AbstractFFTs API

We welcome issues or PRs for functionality not on this list.

Note that some operations not on this list will work, but be slow, due to Base's generic implementations. This is intentional, to enable a "make it work, then make it fast" workflow. When you're ready you can disable slow fallback methods:

```julia
julia> CuArrays.allowscalar(false)
julia> xs[5]
ERROR: getindex is disabled
```

## Current Limitations

When broadcasting, watch out for errors like:

```julia
julia> sin.(cos.(xs))
ERROR: CUDA error: invalid program counter (code #718, ERROR_INVALID_PC)
```

A current limitation of CUDAnative means that you'll need to restart Julia and use `CUDAnative.sin`, `CUDAnative.cos` etc in this case.

There is currently no support for strided arrays or views, though these are planned.
