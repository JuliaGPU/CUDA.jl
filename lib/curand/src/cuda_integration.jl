# High-level rand/randn/seed! API and integration with CUDACore types

using CUDACore: AnyCuArray, CuArray, CuContext, active_state
using CUDACore: GPUArrays


## native RNG (kernel-based Philox2x32)
#
# Holds no GPU memory, just two UInt32s, so no HandleCache is needed: each
# task constructs its own and TLS-drop / task-GC frees it.

const native_state_cache = CUDACore.TaskLocalCache{CuContext, NativeRNG}(:cuRAND_NativeRNG)

"""
    cuRAND.native_rng() -> NativeRNG

The [`NativeRNG`](@ref) of the current task, for its active context. It is seeded by
[`cuRAND.seed!`](@ref).
"""
function native_rng()
    cuda = active_state()
    get!(() -> NativeRNG(), native_state_cache, cuda.context)
end


## default RNG: task-local cached GPUArrays.RNG{CuArray} (the fast Philox4x32-10
## batched-kernel RNG). Used by Random.rand!/randn!(::AnyCuArray) when no rng
## is supplied, and exposed via CUDA.RNG / CUDA.gpuarrays_rng().

const gpuarrays_state_cache = CUDACore.TaskLocalCache{CuContext, GPUArrays.RNG{CuArray}}(:cuRAND_DefaultRNG)

"""
    cuRAND.gpuarrays_rng() -> GPUArrays.RNG

The GPUArrays.jl random number generator of the current task, for its active context.
`cuRAND.rand` and `randn`, and `rand!` and `randn!` without a generator, use it for the
element types that cuRAND does not support. `CUDA.default_rng()` returns it. It is seeded
randomly when a task first uses it, and by [`cuRAND.seed!`](@ref).
"""
function gpuarrays_rng()
    cuda = active_state()
    get!(gpuarrays_state_cache, cuda.context) do
        new_rng = GPUArrays.RNG{CuArray}()
        Random.seed!(new_rng)
        new_rng
    end
end


## cuRAND.rand / cuRAND.randn / cuRAND.seed! — high-level API

"""
    cuRAND.seed!([seed])

Seed the random number generators of the current task with `seed`, or a random seed: the
ones that `cuRAND.rand` and its relatives use ([`library_rng`](@ref) and
[`gpuarrays_rng`](@ref)), and [`native_rng`](@ref). `CUDA.seed!` is this function.

This is not `Random.seed!`, which seeds the CPU's generator. To seed a single generator,
call `Random.seed!` on it.
"""
function seed!(seed=Base.rand(UInt64))
    Random.seed!(gpuarrays_rng(), seed)
    Random.seed!(native_rng(), seed)
    Random.seed!(library_rng(), seed)
end

# cuRAND in-place (convenience without explicit RNG, for the types cuRAND supports)
Random.rand!(A::UniformArray) = Random.rand!(library_rng(), A)
Random.randn!(A::NormalArray; kwargs...) = Random.randn!(library_rng(), A; kwargs...)
rand_logn!(A::LognormalArray; kwargs...) = rand_logn!(library_rng(), A; kwargs...)
rand_poisson!(A::PoissonArray; kwargs...) = rand_poisson!(library_rng(), A; kwargs...)

# GPUArrays RNG fallback for types not supported by cuRAND
Random.rand!(A::AnyCuArray) = Random.rand!(gpuarrays_rng(), A)
function Random.randn!(A::AnyCuArray; mean=0, stddev=1)
    Random.randn!(gpuarrays_rng(), A)
    if mean != 0 || stddev != 1
        A .= A .* stddev .+ mean
    end
    return A
end
rand_logn!(A::AnyCuArray; kwargs...) =
    error("cuRAND does not support generating lognormally-distributed random numbers of type $(eltype(A))")
rand_poisson!(A::AnyCuArray; kwargs...) =
    error("cuRAND does not support generating Poisson-distributed random numbers of type $(eltype(A))")

# cuRAND out-of-place
rand(T::UniformType, dims::Dims) = Random.rand(library_rng(), T, dims)
randn(T::NormalType, dims::Dims; kwargs...) = Random.randn(library_rng(), T, dims; kwargs...)
rand(T::UniformType, dim1::Integer, dims::Integer...) =
    Random.rand(library_rng(), T, Dims((dim1, dims...)))
randn(T::NormalType, dim1::Integer, dims::Integer...; kwargs...) =
    Random.randn(library_rng(), T, Dims((dim1, dims...)); kwargs...)

# GPUArrays out-of-place (fallback for types not supported by cuRAND)
rand(T::Type, dims::Dims) = Random.rand!(CuArray{T}(undef, dims...))
randn(T::Type, dims::Dims; kwargs...) = Random.randn!(CuArray{T}(undef, dims...); kwargs...)
rand(T::Type, dim1::Integer, dims::Integer...) =
    Random.rand!(CuArray{T}(undef, dim1, dims...))
randn(T::Type, dim1::Integer, dims::Integer...; kwargs...) =
    Random.randn!(CuArray{T}(undef, dim1, dims...); kwargs...)

# untyped out-of-place (defaults to cuRAND Float32)
rand(dim1::Integer, dims::Integer...) =
    Random.rand(library_rng(), Dims((dim1, dims...)))
randn(dim1::Integer, dims::Integer...; kwargs...) =
    Random.randn(library_rng(), Dims((dim1, dims...)); kwargs...)

# out-of-place logn/poisson
rand_logn(T::LognormalType, dims::Dims; kwargs...) = rand_logn(library_rng(), T, dims; kwargs...)
rand_poisson(T::PoissonType, dims::Dims; kwargs...) = rand_poisson(library_rng(), T, dims; kwargs...)
rand_logn(T::LognormalType, dim1::Integer, dims::Integer...; kwargs...) =
    rand_logn(library_rng(), T, Dims((dim1, dims...)); kwargs...)
rand_poisson(T::PoissonType, dim1::Integer, dims::Integer...; kwargs...) =
    rand_poisson(library_rng(), T, Dims((dim1, dims...)); kwargs...)
rand_logn(dim1::Integer, dims::Integer...; kwargs...) =
    rand_logn(library_rng(), Dims((dim1, dims...)); kwargs...)
rand_poisson(dim1::Integer, dims::Integer...; kwargs...) =
    rand_poisson(library_rng(), Dims((dim1, dims...)); kwargs...)

# scalars
rand(T::Type=Float32) = rand(T, 1)[]
randn(T::Type=Float32; kwargs...) = randn(T, 1; kwargs...)[]
rand_logn(T::Type=Float32; kwargs...) = rand_logn(library_rng(), T, 1; kwargs...)[]
rand_poisson(T::Type=Cuint; kwargs...) = rand_poisson(library_rng(), T, 1; kwargs...)[]


## documentation

"""
    cuRAND.rand([T=Float32,] dims...) -> CuArray{T}
    cuRAND.rand(T=Float32) -> T

Generate a `CuArray` of uniformly distributed random numbers, or a single number, on the
GPU. `Float32`, `Float64` and `UInt32` numbers come from the task's cuRAND generator
([`library_rng`](@ref)), other types from its GPUArrays generator
([`gpuarrays_rng`](@ref)). `CUDA.rand` is this function. A single number is
read back from the GPU by indexing, which needs `CUDA.@allowscalar` outside interactive
sessions.

This is not `Random.rand`, which generates numbers on the CPU, but it takes the same
arguments without the generator. To use a specific generator, call `Random.rand` with a
[`LibraryRNG`](@ref) or [`NativeRNG`](@ref).
"""
rand

"""
    cuRAND.randn([T=Float32,] dims...; mean=0, stddev=1) -> CuArray{T}
    cuRAND.randn(T=Float32; mean=0, stddev=1) -> T

Generate a `CuArray` of normally distributed random numbers with mean `mean` and standard
deviation `stddev`, or a single number, on the GPU. `Float32` and `Float64` numbers come
from the task's cuRAND generator ([`library_rng`](@ref)), other types from its GPUArrays
generator ([`gpuarrays_rng`](@ref)). `CUDA.randn` is this function. A single
number is read back from the GPU by indexing, which needs `CUDA.@allowscalar` outside
interactive sessions.

This is not `Random.randn`, which generates numbers on the CPU, but it takes the same
arguments without the generator. To use a specific generator, call `Random.randn` with a
[`LibraryRNG`](@ref) or [`NativeRNG`](@ref).
"""
randn

"""
    cuRAND.rand_logn([rng::LibraryRNG,] [T=Float32,] dims...; mean=0, stddev=1) -> CuArray{T}
    cuRAND.rand_logn([rng::LibraryRNG,] T=Float32; mean=0, stddev=1) -> T

Generate a `CuArray` of log-normally distributed random numbers, or a single number, with
cuRAND: `exp(x)` for normally distributed `x` with mean `mean` and standard deviation
`stddev`. `T` is `Float32` or `Float64`. Without `rng`, the task's [`library_rng`](@ref) is
used. See also [`rand_logn!`](@ref). A single number is read back from the GPU by
indexing, which needs `CUDA.@allowscalar` outside interactive sessions.
"""
rand_logn

"""
    cuRAND.rand_poisson([rng::LibraryRNG,] [T=UInt32,] dims...; lambda=1) -> CuArray{T}
    cuRAND.rand_poisson([rng::LibraryRNG,] T=UInt32; lambda=1) -> T

Generate a `CuArray` of Poisson-distributed random numbers with mean `lambda`, or a single
number, with cuRAND. `T` must be `UInt32`. Without `rng`, the task's
[`library_rng`](@ref) is used. See also [`rand_poisson!`](@ref). A single number is read
back from the GPU by indexing, which needs `CUDA.@allowscalar` outside interactive
sessions.
"""
rand_poisson
