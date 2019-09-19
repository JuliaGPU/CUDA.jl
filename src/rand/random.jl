# integration with Random

using Random

using GPUArrays

export rand_logn!, rand_poisson!

# the interface is split in two levels:
# - functions that extend the Random standard library, and take an RNG as first argument,
#   will only ever dispatch to CURAND and as a result are limited in the types they support.
# - functions that take an array will dispatch to either CURAND or GPUArrays
# - non-exported functions are provided for constructing GPU arrays from only an eltype


mutable struct RNG <: Random.AbstractRNG
    ptr::curandGenerator_t
    typ::Int

    function RNG(typ=CURAND_RNG_PSEUDO_DEFAULT)
        ptr = Ref{curandGenerator_t}()
        curandCreateGenerator(ptr, typ)
        obj = new(ptr[], typ)
        finalizer(curandDestroyGenerator, obj)
        return obj
    end
end

Base.unsafe_convert(::Type{curandGenerator_t}, rng::RNG) = rng.ptr


## seeding

seed!(rng::RNG=generator()) = (curandGenerateSeeds(rng); return)

seed!(seed::Int64, offset::Int64=0) = seed!(generator(), seed, offset)
function seed!(rng::RNG, seed::Int64, offset::Int64)
    curandSetPseudoRandomGeneratorSeed(rng, seed)
    curandSetGeneratorOffset(rng, offset)
    curandGenerateSeeds(rng)
    return
end


## in-place

# uniform
const UniformType = Union{Type{Float32},Type{Float64}}
const UniformArray = CuArray{<:Union{Float32,Float64}}
function Random.rand!(rng::RNG, A::CuArray{Float32})
    curandGenerateUniform(rng, A, length(A))
    return A
end
function Random.rand!(rng::RNG, A::CuArray{Float64})
    curandGenerateUniformDouble(rng, A, length(A))
    return A
end

# some functions need pow2 lengths: use a padded array and copy back to the original one
function inplace_pow2(A, f)
    len = length(A)
    if len > 1 && ispow2(len)
        f(A)
    else
        padlen = max(2, nextpow(2, len))
        B = similar(A, padlen)
        f(B)
        copyto!(A, 1, B, 1, len)
        CuArrays.unsafe_free!(B)
    end
    A
end

# normal
const NormalType = Union{Type{Float32},Type{Float64}}
const NormalArray = CuArray{<:Union{Float32,Float64}}
function Random.randn!(rng::RNG, A::CuArray{Float32}; mean=0, stddev=1)
    inplace_pow2(A, B->curandGenerateNormal(rng, B, length(B), mean, stddev))
    return A
end
function Random.randn!(rng::RNG, A::CuArray{Float64}; mean=0, stddev=1)
    inplace_pow2(A, B->curandGenerateNormalDouble(rng, B, length(B), mean, stddev))
    return A
end

# log-normal
const LognormalType = Union{Type{Float32},Type{Float64}}
const LognormalArray = CuArray{<:Union{Float32,Float64}}
function rand_logn!(rng::RNG, A::CuArray{Float32}; mean=0, stddev=1)
    inplace_pow2(A, B->curandGenerateLogNormal(rng, B, length(B), mean, stddev))
    return A
end
function rand_logn!(rng::RNG, A::CuArray{Float64}; mean=0, stddev=1)
    inplace_pow2(A, B->curandGenerateLogNormalDouble(rng, B, length(B), mean, stddev))
    return A
end

# poisson
const PoissonType = Union{Type{Cuint}}
const PoissonArray = CuArray{Cuint}
function rand_poisson!(rng::RNG, A::CuArray{Cuint}; lambda=1)
    curandGeneratePoisson(rng, A, length(A), lambda)
    return A
end


## out of place

# some functions need pow2 lengths: construct a compatible array and return part of it
function outofplace_pow2(shape, ctor, f)
    len = prod(shape)
    if ispow2(len)
        A = ctor(shape)
        f(A)
    else
        padlen = max(2, nextpow(2, len))
        A = ctor(padlen)
        f(A)
        B = reshape(A[1:len], shape)
        return B
    end
end

# arrays
Random.rand(rng::RNG, T::UniformType, dims::Dims) =
    Random.rand!(rng, CuArray{T}(undef, dims))
Random.randn(rng::RNG, T::NormalType, dims::Dims; kwargs...) =
    outofplace_pow2(dims, shape->CuArray{T}(undef, dims), A->randn!(rng, A; kwargs...))
rand_logn(rng::RNG, T::LognormalType, dims::Dims; kwargs...) =
    outofplace_pow2(dims, shape->CuArray{T}(undef, dims), A->rand_logn!(rng, A; kwargs...))
rand_poisson(rng::RNG, T::PoissonType, dims::Dims; kwargs...) =
    rand_poisson!(rng, CuArray{T}(undef, dims); kwargs...)

# specify default types
Random.rand(rng::RNG, dims::Dims; kwargs...) = Random.rand(rng, Float32, dims; kwargs...)
Random.randn(rng::RNG, dims::Dims; kwargs...) = Random.randn(rng, Float32, dims; kwargs...)
rand_logn(rng::RNG, dims::Dims; kwargs...) = rand_logn(rng, Float32, dims; kwargs...)
rand_poisson(rng::RNG, dims::Dims; kwargs...) = rand_poisson(rng, Cuint, dims; kwargs...)

# support all dimension specifications
Random.rand(rng::RNG, T::UniformType, dim1::Integer, dims::Integer...) =
    Random.rand(rng, T, Dims((dim1, dims...)))
Random.randn(rng::RNG, T::NormalType, dim1::Integer, dims::Integer...; kwargs...) =
    Random.randn(rng, T, Dims((dim1, dims...)); kwargs...)
rand_logn(rng::RNG, T::LognormalType, dim1::Integer, dims::Integer...; kwargs...) =
    rand_logn(rng, T, Dims((dim1, dims...)); kwargs...)
rand_poisson(rng::RNG, T::PoissonType, dim1::Integer, dims::Integer...; kwargs...) =
    rand_poisson(rng, T, Dims((dim1, dims...)); kwargs...)


## functions that dispatch to either CURAND or GPUArrays

# CURAND in-place
Random.rand!(A::UniformArray) = Random.rand!(generator(), A)
Random.randn!(A::NormalArray; kwargs...) = Random.randn!(generator(), A; kwargs...)
rand_logn!(A::LognormalArray; kwargs...) = rand_logn!(generator(), A; kwargs...)
rand_poisson!(A::PoissonArray; kwargs...) = rand_poisson!(generator(), A; kwargs...)

# CURAND out-of-place
rand(T::UniformType, dims::Dims) = Random.rand(generator(), T, dims)
randn(T::NormalType, dims::Dims; kwargs...) = Random.randn(generator(), T, dims; kwargs...)
rand_logn(T::LognormalType, dims::Dims; kwargs...) = rand_logn(generator(), T, dims; kwargs...)
rand_poisson(T::PoissonType, dims::Dims; kwargs...) = rand_poisson(generator(), T, dims; kwargs...)

# support all dimension specifications
rand(T::UniformType, dim1::Integer, dims::Integer...) =
    Random.rand(generator(), T, Dims((dim1, dims...)))
randn(T::NormalType, dim1::Integer, dims::Integer...; kwargs...) =
    Random.randn(generator(), T, Dims((dim1, dims...)); kwargs...)
rand_logn(T::LognormalType, dim1::Integer, dims::Integer...; kwargs...) =
    rand_logn(generator(), T, Dims((dim1, dims...)); kwargs...)
rand_poisson(T::PoissonType, dim1::Integer, dims::Integer...; kwargs...) =
    rand_poisson(generator(), T, Dims((dim1, dims...)); kwargs...)

# GPUArrays in-place
Random.rand!(A::CuArray) = Random.rand!(GPUArrays.global_rng(A), A)
Random.randn!(A::CuArray; kwargs...) =
    error("CuArrays does not support generating normally-distrubyted random numbers of type $(eltype(A))")
rand_logn!(A::CuArray; kwargs...) =
    error("CuArrays does not support generating lognormally-distributed random numbers of type $(eltype(A))")
rand_poisson!(A::CuArray; kwargs...) =
    error("CuArrays does not support generating Poisson-distributed random numbers of type $(eltype(A))")

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
    Random.rand(generator(), Dims((dim1, dims...)))
randn(dim1::Integer, dims::Integer...; kwargs...) =
    Random.randn(generator(), Dims((dim1, dims...)); kwargs...)
rand_logn(dim1::Integer, dims::Integer...; kwargs...) =
    rand_logn(generator(), Dims((dim1, dims...)); kwargs...)
rand_poisson(dim1::Integer, dims::Integer...; kwargs...) =
    rand_poisson(generator(), Dims((dim1, dims...)); kwargs...)
