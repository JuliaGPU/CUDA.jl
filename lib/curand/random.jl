# interfacing with Random standard library

using Random

using GPUArrays


mutable struct RNG <: Random.AbstractRNG
    handle::curandGenerator_t
    ctx::CuContext
    typ::Int

    function RNG(typ=CURAND_RNG_PSEUDO_DEFAULT)
        handle = curandCreateGenerator(typ)
        curandSetStream(handle, stream())   # XXX: duplicate with default_rng
        obj = new(handle, context(), typ)
        finalizer(unsafe_destroy!, obj)
        return obj
    end

    # TODO: this design doesn't work nicely in the presence of streams.
    #       if the user switches streams, a local RNG object won't be updated.
end

function unsafe_destroy!(rng::RNG)
    CUDA.isvalid(rng.ctx) || return
    context!(rng.ctx) do
        curandDestroyGenerator(rng)
    end
end

Base.unsafe_convert(::Type{curandGenerator_t}, rng::RNG) = rng.handle


## seeding

function Random.seed!(rng::RNG, seed=Base.rand(UInt64), offset=0)
    curandSetPseudoRandomGeneratorSeed(rng, seed)
    curandSetGeneratorOffset(rng, offset)
    res = @retry_reclaim err->isequal(err, CURAND_STATUS_ALLOCATION_FAILED) ||
                              isequal(err, CURAND_STATUS_PREEXISTING_FAILURE) begin
        unsafe_curandGenerateSeeds(rng)
    end
    if res != CURAND_STATUS_SUCCESS
        throw_api_error(res)
    end
    return
end

Random.seed!(rng::RNG, ::Nothing) = Random.seed!(rng)


## in-place

# uniform
const UniformType = Union{Type{Float32},Type{Float64},Type{UInt32}}
const UniformArray = DenseCuArray{<:Union{Float32,Float64,UInt32}}
function Random.rand!(rng::RNG, A::DenseCuArray{UInt32})
    curandGenerate(rng, A, length(A))
    return A
end
function Random.rand!(rng::RNG, A::DenseCuArray{Float32})
    curandGenerateUniform(rng, A, length(A))
    return A
end
function Random.rand!(rng::RNG, A::DenseCuArray{Float64})
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
        CUDA.unsafe_free!(B)
    end
    A
end

# normal
const NormalType = Union{Type{Float32},Type{Float64}}
const NormalArray = DenseCuArray{<:Union{Float32,Float64}}
function Random.randn!(rng::RNG, A::DenseCuArray{Float32}; mean=0, stddev=1)
    inplace_pow2(A, B->curandGenerateNormal(rng, B, length(B), mean, stddev))
    return A
end
function Random.randn!(rng::RNG, A::DenseCuArray{Float64}; mean=0, stddev=1)
    inplace_pow2(A, B->curandGenerateNormalDouble(rng, B, length(B), mean, stddev))
    return A
end

# log-normal
const LognormalType = Union{Type{Float32},Type{Float64}}
const LognormalArray = DenseCuArray{<:Union{Float32,Float64}}
function rand_logn!(rng::RNG, A::DenseCuArray{Float32}; mean=0, stddev=1)
    inplace_pow2(A, B->curandGenerateLogNormal(rng, B, length(B), mean, stddev))
    return A
end
function rand_logn!(rng::RNG, A::DenseCuArray{Float64}; mean=0, stddev=1)
    inplace_pow2(A, B->curandGenerateLogNormalDouble(rng, B, length(B), mean, stddev))
    return A
end

# poisson
const PoissonType = Union{Type{Cuint}}
const PoissonArray = DenseCuArray{Cuint}
function rand_poisson!(rng::RNG, A::DenseCuArray{Cuint}; lambda=1)
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

