# CuArrays

CuArrays provides a fully-functional GPU array, backed by [CUDAnative](https://github.com/JuliaGPU/CUDAnative.jl). CuArrays may give significant speedups over CPU arrays without code changes, though the package is a work in progress.

```julia
Pkg.add("CUDAnative") # see instructions
Pkg.add("CUBLAS")
Pkg.clone("https://github.com/MikeInnes/CuArrays.jl")
```

Note that some tests currently fail under `Pkg.test()`; you can run `julia --color=yes test/runtests.jl` to check your installation instead.

## Features

```julia
xs = CuArray(rand(5, 5))
ys = cu[1, 2, 3]
xs_cpu = collect(xs)
```

Because `CuArray` is an `AbstractArray`, it doesn't have much of a learning curve; just use your favourite array ops as usual. The following are supported (on arbitrary numbers of arguments, dimensions etc):

* Conversions and `copy!` with CPU arrays
* General indexing (`xs[1:2, 5, :]`)
* `permutedims`
* Concatenation (`vcat(x, y)`, `cat(3, xs, ys, zs)`)
* `map`, fused broadcast (`zs .= sin.(xs) .+ ys .* 2`)
* `fill!(xs, 0)`
* Reduction over dimensions (`reducedim(+, xs, 3)`, `sum(x -> x^2, xs, 1)` etc)
* Reduction to scalar (`reduce(*, 1, xs)`, `sum(xs)` etc)
* Various BLAS operations (matrix\*matrix, matrix\*vector)

We welcome issues or PRs for functionality not on this list.

Note that some operations not on this list will work, but be slow, due to Base's generic implementations. This is intentional, to enable a "make it work, then make it fast" workflow. When you're ready you can disable slow fallback methods:

```julia
julia> CuArrays.allowslow(false)
julia> xs[5]
ERROR: getindex is disabled
```

## Current Limitations

When broadcasting, watch out for errors like:

```julia
julia> sin.(cos.(xs))
ERROR: CUDA error: invalid program counter (code #718, ERROR_INVALID_PC)
```

A current limitation of CUDAnative means that you'll need to restart Julia and use `CuArrays.sin`, `CuArrays.cos` etc in this case.

There is currently no support for strided arrays or views, though these are planned.
