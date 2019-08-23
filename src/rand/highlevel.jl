# high-level interface for CURAND
#
# the interface is split in two levels:
# - functions that extend the Random standard library, and take an RNG as first argument,
#   will only ever dispatch to CURAND and as a result are limited in the types they support.
# - functions that take an array will dispatch to either CURAND or GPUArrays
# - `cu`-prefixed functions are provided for constructing GPU arrays from only an eltype


## seeding

seed!(rng::RNG=generator()) = curandGenerateSeeds(rng)


## in-place

# uniform
const UniformTypes = Union{Float32,Float64}
Random.rand!(rng::RNG, A::CuArray{Float32}) = curandGenerateUniform(rng, A)
Random.rand!(rng::RNG, A::CuArray{Float64}) = curandGenerateUniformDouble(rng, A)

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
const NormalTypes = Union{Float32,Float64}
Random.randn!(rng::RNG, A::CuArray{Float32}; mean=0, stddev=1) = inplace_pow2(A, B->curandGenerateNormal(rng, B, mean, stddev))
Random.randn!(rng::RNG, A::CuArray{Float64}; mean=0, stddev=1) = inplace_pow2(A, B->curandGenerateNormalDouble(rng, B, mean, stddev))

# log-normal
const LognormalTypes = Union{Float32,Float64}
rand_logn!(rng::RNG, A::CuArray{Float32}; mean=0, stddev=1) = inplace_pow2(A, B->curandGenerateLogNormal(rng, B, mean, stddev))
rand_logn!(rng::RNG, A::CuArray{Float64}; mean=0, stddev=1) = inplace_pow2(A, B->curandGenerateLogNormalDouble(rng, B, mean, stddev))

# poisson
const PoissonTypes = Union{Cuint}
rand_poisson!(rng::RNG, A::CuArray{Cuint}; lambda=1) = curandGeneratePoisson(rng, A, lambda)


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

Random.rand(rng::RNG, ::Type{X}, dims::Dims) where {X} =
    Random.rand!(rng, CuArray{X}(undef, dims))
Random.randn(rng::RNG, ::Type{X}, dims::Dims; kwargs...) where {X} =
    outofplace_pow2(dims, shape->CuArray{X}(undef, dims), A->randn!(rng, A; kwargs...))
rand_logn(rng::RNG, ::Type{X}, dims::Dims; kwargs...) where {X} =
    outofplace_pow2(dims, shape->CuArray{X}(undef, dims), A->rand_logn!(rng, A; kwargs...))
rand_poisson(rng::RNG, ::Type{X}, dims::Dims; kwargs...) where {X} =
    rand_poisson!(rng, CuArray{X}(undef, dims); kwargs...)

# specify default types
Random.rand(rng::RNG, dims::Dims; kwargs...) = Random.rand(rng, Float32, dims; kwargs...)
Random.randn(rng::RNG, dims::Dims; kwargs...) = Random.randn(rng, Float32, dims; kwargs...)
rand_logn(rng::RNG, dims::Dims; kwargs...) = rand_logn(rng, Float32, dims; kwargs...)
rand_poisson(rng::RNG, dims::Dims; kwargs...) = rand_poisson(rng, Cuint, dims; kwargs...)

# support all dimension specifications
Random.rand(rng::RNG, ::Type{X}, dim1::Integer, dims::Integer...) where {X} =
    Random.rand(rng, X, Dims((dim1, dims...)))
Random.randn(rng::RNG, ::Type{X}, dim1::Integer, dims::Integer...; kwargs...) where {X} =
    Random.randn(rng, X, Dims((dim1, dims...)); kwargs...)
rand_logn(rng::RNG, ::Type{X}, dim1::Integer, dims::Integer...; kwargs...) where {X} =
    rand_logn(rng, X, Dims((dim1, dims...)); kwargs...)
rand_poisson(rng::RNG, ::Type{X}, dim1::Integer, dims::Integer...; kwargs...) where {X} =
    rand_poisson(rng, X, Dims((dim1, dims...)); kwargs...)


## functions that dispatch to either CURAND or GPUArrays

# CURAND in-place
Random.rand!(A::CuArray{X}) where {X <: UniformTypes} = Random.rand!(generator(), A)
Random.randn!(A::CuArray{X}; kwargs...) where {X <: NormalTypes} = Random.randn!(generator(), A; kwargs...)
rand_logn!(A::CuArray{X}; kwargs...) where {X <: LognormalTypes} = rand_logn!(generator(), A; kwargs...)
rand_poisson!(A::CuArray{X}; kwargs...) where {X <: PoissonTypes} = rand_poisson!(generator(), A; kwargs...)

# CURAND out-of-place
rand(::Type{X}, dims::Dims) where {X <: UniformTypes} = Random.rand(generator(), X, dims)
randn(::Type{X}, dims::Dims; kwargs...) where {X <: NormalTypes} = Random.randn(generator(), X, dims; kwargs...)
rand_logn(rng::Type{X}, dims::Dims; kwargs...) where {X <: LognormalTypes} = rand_logn(generator(), X, dims; kwargs...)
rand_poisson(rng::Type{X}, dims::Dims; kwargs...) where {X <: PoissonTypes} = rand_poisson(generator(), X, dims; kwargs...)

# support all dimension specifications
rand(::Type{X}, dim1::Integer, dims::Integer...) where {X <: UniformTypes} =
    Random.rand(generator(), X, Dims((dim1, dims...)))
randn(::Type{X}, dim1::Integer, dims::Integer...; kwargs...) where {X <: NormalTypes} =
    Random.randn(generator(), X, Dims((dim1, dims...)); kwargs...)
rand_logn(::Type{X}, dim1::Integer, dims::Integer...; kwargs...) where {X <: LognormalTypes} =
    rand_logn(generator(), X, Dims((dim1, dims...)); kwargs...)
rand_poisson(::Type{X}, dim1::Integer, dims::Integer...; kwargs...) where {X <: PoissonTypes} =
    rand_poisson(generator(), X, Dims((dim1, dims...)); kwargs...)

# GPUArrays in-place
Random.rand!(A::CuArray{X}) where {X} = Random.rand!(GPUArrays.global_rng(A), A)
Random.randn!(A::CuArray{X}; kwargs...) where {X} = error("CuArrays does not support generating normally-distrubyted random numbers of type $X")
rand_logn!(A::CuArray{X}; kwargs...) where {X} = error("CuArrays does not support generating lognormally-distributed random numbers of type $X")
rand_poisson!(A::CuArray{X}; kwargs...) where {X} = error("CuArrays does not support generating Poisson-distributed random numbers of type $X")

# GPUArrays out-of-place
rand(::Type{X}, dims::Dims) where {X} = Random.rand!(CuArray{X}(undef, dims...))
randn(::Type{X}, dims::Dims; kwargs...) where {X} = Random.randn!(CuArray{X}(undef, dims...); kwargs...)
rand_logn(rng::Type{X}, dims::Dims; kwargs...) where {X} = rand_logn!(CuArray{X}(undef, dims...); kwargs...)
rand_poisson(rng::Type{X}, dims::Dims; kwargs...) where {X} = rand_poisson!(CuArray{X}(undef, dims...); kwargs...)

# support all dimension specifications
rand(::Type{X}, dim1::Integer, dims::Integer...) where {X} =
    Random.rand!(CuArray{X}(undef, dim1, dims...))
randn(::Type{X}, dim1::Integer, dims::Integer...; kwargs...) where {X} =
    Random.randn!(CuArray{X}(undef, dim1, dims...); kwargs...)
rand_logn(::Type{X}, dim1::Integer, dims::Integer...; kwargs...) where {X} =
    rand_logn!(CuArray{X}(undef, dim1, dims...); kwargs...)
rand_poisson(::Type{X}, dim1::Integer, dims::Integer...; kwargs...) where {X} =
    rand_poisson!(CuArray{X}(undef, dim1, dims...); kwargs...)

# untyped out-of-place
rand(dim1::Integer, dims::Integer...) =
    Random.rand(generator(), Dims((dim1, dims...)))
randn(dim1::Integer, dims::Integer...; kwargs...) =
    Random.randn(generator(), Dims((dim1, dims...)); kwargs...)
rand_logn(dim1::Integer, dims::Integer...; kwargs...) =
    rand_logn(generator(), Dims((dim1, dims...)); kwargs...)
rand_poisson(dim1::Integer, dims::Integer...; kwargs...) =
    rand_poisson(generator(), Dims((dim1, dims...)); kwargs...)
