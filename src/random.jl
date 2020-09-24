# random functions that dispatch either to CURAND or GPUArrays' generic RNG

using Random

export rand_logn!, rand_poisson!

# the interface is split in two levels:
# - functions that extend the Random standard library, and take an RNG as first argument,
#   will only ever dispatch to CURAND and as a result are limited in the types they support.
# - functions that take an array will dispatch to either CURAND or GPUArrays
# - non-exported functions are provided for constructing GPU arrays from only an eltype

const curand_rng = CURAND.default_rng
gpuarrays_rng() = GPUArrays.default_rng(CuArray)

function seed!(seed=Base.rand(UInt64))
    Random.seed!(curand_rng(), seed)
    Random.seed!(gpuarrays_rng(), seed)
end

# CURAND in-place
Random.rand!(A::CURAND.UniformArray) = Random.rand!(curand_rng(), A)
Random.randn!(A::CURAND.NormalArray; kwargs...) = Random.randn!(curand_rng(), A; kwargs...)
rand_logn!(A::CURAND.LognormalArray; kwargs...) = CURAND.rand_logn!(curand_rng(), A; kwargs...)
rand_poisson!(A::CURAND.PoissonArray; kwargs...) = CURAND.rand_poisson!(curand_rng(), A; kwargs...)

# CURAND out-of-place
rand(T::CURAND.UniformType, dims::Dims) = Random.rand(curand_rng(), T, dims)
randn(T::CURAND.NormalType, dims::Dims; kwargs...) = Random.randn(curand_rng(), T, dims; kwargs...)
rand_logn(T::CURAND.LognormalType, dims::Dims; kwargs...) = CURAND.rand_logn(curand_rng(), T, dims; kwargs...)
rand_poisson(T::CURAND.PoissonType, dims::Dims; kwargs...) = CURAND.rand_poisson(curand_rng(), T, dims; kwargs...)

# support all dimension specifications
rand(T::CURAND.UniformType, dim1::Integer, dims::Integer...) =
    Random.rand(curand_rng(), T, Dims((dim1, dims...)))
randn(T::CURAND.NormalType, dim1::Integer, dims::Integer...; kwargs...) =
    Random.randn(curand_rng(), T, Dims((dim1, dims...)); kwargs...)
rand_logn(T::CURAND.LognormalType, dim1::Integer, dims::Integer...; kwargs...) =
    CURAND.rand_logn(curand_rng(), T, Dims((dim1, dims...)); kwargs...)
rand_poisson(T::CURAND.PoissonType, dim1::Integer, dims::Integer...; kwargs...) =
    CURAND.rand_poisson(curand_rng(), T, Dims((dim1, dims...)); kwargs...)

# GPUArrays in-place
Random.rand!(A::AnyCuArray) = Random.rand!(gpuarrays_rng(), A)
Random.randn!(A::AnyCuArray; kwargs...) =
    error("CUDA.jl does not support generating normally-distributed random numbers of type $(eltype(A))")
# FIXME: GPUArrays.jl has a randn! nowadays, but it doesn't work with e.g. Cuint
rand_logn!(A::AnyCuArray; kwargs...) =
    error("CUDA.jl does not support generating lognormally-distributed random numbers of type $(eltype(A))")
rand_poisson!(A::AnyCuArray; kwargs...) =
    error("CUDA.jl does not support generating Poisson-distributed random numbers of type $(eltype(A))")

# GPUArrays out-of-place
rand(T::Type, dims::Dims) = Random.rand!(CuArray{T}(undef, dims...))
randn(T::Type, dims::Dims; kwargs...) = Random.randn!(CuArray{T}(undef, dims...); kwargs...)
rand_logn(T::Type, dims::Dims; kwargs...) = rand_logn!(CuArray{T}(undef, dims...); kwargs...)
rand_poisson(T::Type, dims::Dims; kwargs...) = rand_poisson!(CuArray{T}(undef, dims...); kwargs...)

# support all dimension specifications
rand(T::Type, dim1::Integer, dims::Integer...) =
    Random.rand!(CuArray{T}(undef, dim1, dims...))
randn(T::Type, dim1::Integer, dims::Integer...; kwargs...) =
    Random.randn!(CuArray{T}(undef, dim1, dims...); kwargs...)
rand_logn(T::Type, dim1::Integer, dims::Integer...; kwargs...) =
    rand_logn!(CuArray{T}(undef, dim1, dims...); kwargs...)
rand_poisson(T::Type, dim1::Integer, dims::Integer...; kwargs...) =
    rand_poisson!(CuArray{T}(undef, dim1, dims...); kwargs...)

# untyped out-of-place
rand(dim1::Integer, dims::Integer...) =
    Random.rand(curand_rng(), Dims((dim1, dims...)))
randn(dim1::Integer, dims::Integer...; kwargs...) =
    Random.randn(curand_rng(), Dims((dim1, dims...)); kwargs...)
rand_logn(dim1::Integer, dims::Integer...; kwargs...) =
    CURAND.rand_logn(curand_rng(), Dims((dim1, dims...)); kwargs...)
rand_poisson(dim1::Integer, dims::Integer...; kwargs...) =
    CURAND.rand_poisson(curand_rng(), Dims((dim1, dims...)); kwargs...)
