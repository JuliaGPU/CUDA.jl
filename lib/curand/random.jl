# interfacing with Random standard library

using Random


mutable struct RNG <: Random.AbstractRNG
    handle::curandGenerator_t
    ctx::CuContext
    stream::CuStream
    typ::Int

    function RNG(typ=CURAND_RNG_PSEUDO_DEFAULT; stream=stream())
        handle = curandCreateGenerator(typ)
        if stream !== nothing
            curandSetStream(handle, stream)
        end
        obj = new(handle, context(), stream, typ)
        finalizer(unsafe_destroy!, obj)
        return obj
    end
end

function unsafe_destroy!(rng::RNG)
    context!(rng.ctx; skip_destroyed=true) do
        curandDestroyGenerator(rng)
    end
end

Base.unsafe_convert(::Type{curandGenerator_t}, rng::RNG) = rng.handle

# RNG objects can be user-created on a different task, whose stream might be different from
# the one used in the current task. call this function before every API call that performs
# operations on a stream to ensure the RNG is using the correct task-local stream.
@inline function update_stream(rng::RNG)
    new_stream = stream()
    if rng.stream != new_stream
        rng.stream = new_stream
        curandSetStream(rng, new_stream)
    end
    return
end


## seeding

make_seed() = Base.rand(RandomDevice(), UInt64)

function Random.seed!(rng::RNG, seed=make_seed(), offset=0)
    update_stream(rng)
    curandSetPseudoRandomGeneratorSeed(rng, seed)
    curandSetGeneratorOffset(rng, offset)
    curandGenerateSeeds(rng)
    return
end

Random.seed!(rng::RNG, ::Nothing) = Random.seed!(rng)


## in-place

# uniform
const UniformType = Union{Type{Float32},Type{Float64},Type{UInt32}}
const UniformArray = DenseCuArray{<:Union{Float32,Float64,UInt32}}
function Random.rand!(rng::RNG, A::DenseCuArray{UInt32})
    isempty(A) && return A
    update_stream(rng)
    curandGenerate(rng, A, length(A))
    return A
end
function Random.rand!(rng::RNG, A::DenseCuArray{Float32})
    isempty(A) && return A
    update_stream(rng)
    curandGenerateUniform(rng, A, length(A))
    return A
end
function Random.rand!(rng::RNG, A::DenseCuArray{Float64})
    isempty(A) && return A
    update_stream(rng)
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
    isempty(A) && return A
    update_stream(rng)
    inplace_pow2(A, B->curandGenerateNormal(rng, B, length(B), mean, stddev))
    return A
end
function Random.randn!(rng::RNG, A::DenseCuArray{Float64}; mean=0, stddev=1)
    isempty(A) && return A
    update_stream(rng)
    inplace_pow2(A, B->curandGenerateNormalDouble(rng, B, length(B), mean, stddev))
    return A
end

# log-normal
const LognormalType = Union{Type{Float32},Type{Float64}}
const LognormalArray = DenseCuArray{<:Union{Float32,Float64}}
function rand_logn!(rng::RNG, A::DenseCuArray{Float32}; mean=0, stddev=1)
    isempty(A) && return A
    update_stream(rng)
    inplace_pow2(A, B->curandGenerateLogNormal(rng, B, length(B), mean, stddev))
    return A
end
function rand_logn!(rng::RNG, A::DenseCuArray{Float64}; mean=0, stddev=1)
    isempty(A) && return A
    update_stream(rng)
    inplace_pow2(A, B->curandGenerateLogNormalDouble(rng, B, length(B), mean, stddev))
    return A
end

# poisson
const PoissonType = Union{Type{Cuint}}
const PoissonArray = DenseCuArray{Cuint}
function rand_poisson!(rng::RNG, A::DenseCuArray{Cuint}; lambda=1)
    isempty(A) && return A
    update_stream(rng)
    curandGeneratePoisson(rng, A, length(A), lambda)
    return A
end

# CPU arrays
function Random.rand!(rng::RNG, A::AbstractArray{T}) where {T <: UniformType}
    isempty(A) && return A
    B = CuArray{T}(undef, size(A))
    rand!(rng, B)
    copyto!(A, B)
end
function Random.randn!(rng::RNG, A::AbstractArray{T}) where {T <: NormalType}
    isempty(A) && return A
    B = CuArray{T}(undef, size(A))
    randn!(rng, B)
    copyto!(A, B)
end
function rand_logn!(rng::RNG, A::AbstractArray{T}) where {T <: LognormalType}
    isempty(A) && return A
    B = CuArray{T}(undef, size(A))
    rand_logn!(rng, B)
    copyto!(A, B)
end
function rand_poisson!(rng::RNG, A::AbstractArray{T}) where {T <: PoissonType}
    isempty(A) && return A
    B = CuArray{T}(undef, size(A))
    rand_poisson!(rng, B)
    copyto!(A, B)
end


## out of place

# some functions need pow2 lengths: construct a compatible array and return part of it
function outofplace_pow2(shape, ctor, f)
    len = prod(shape)
    if len == 0 || ispow2(len)
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

# GPU arrays
Random.rand(rng::RNG, T::UniformType, dims::Dims) =
    Random.rand!(rng, CuArray{T}(undef, dims))
Random.randn(rng::RNG, T::NormalType, dims::Dims; kwargs...) =
    outofplace_pow2(dims, shape->CuArray{T}(undef, dims), A->randn!(rng, A; kwargs...))
rand_logn(rng::RNG, T::LognormalType, dims::Dims; kwargs...) =
    outofplace_pow2(dims, shape->CuArray{T}(undef, dims), A->rand_logn!(rng, A; kwargs...))
rand_poisson(rng::RNG, T::PoissonType, dims::Dims; kwargs...) =
    rand_poisson!(rng, CuArray{T}(undef, dims); kwargs...)

# specify default types
Random.rand(rng::RNG, dims::Dims; kwargs...) = rand(rng, Float32, dims; kwargs...)
Random.randn(rng::RNG, dims::Dims; kwargs...) = randn(rng, Float32, dims; kwargs...)
rand_logn(rng::RNG, dims::Dims; kwargs...) = rand_logn(rng, Float32, dims; kwargs...)
rand_poisson(rng::RNG, dims::Dims; kwargs...) = rand_poisson(rng, Cuint, dims; kwargs...)

# support all dimension specifications
Random.rand(rng::RNG, dim1::Integer, dims::Integer...) =
    Random.rand(rng, Dims((dim1, dims...)))
Random.randn(rng::RNG, dim1::Integer, dims::Integer...; kwargs...) =
    Random.randn(rng, Dims((dim1, dims...)); kwargs...)
rand_logn(rng::RNG, dim1::Integer, dims::Integer...; kwargs...) =
    rand_logn(rng, Dims((dim1, dims...)); kwargs...)
rand_poisson(rng::RNG, dim1::Integer, dims::Integer...; kwargs...) =
    rand_poisson(rng, Dims((dim1, dims...)); kwargs...)
# ... and with a type
Random.rand(rng::RNG, T::UniformType, dim1::Integer, dims::Integer...) =
    Random.rand(rng, T, Dims((dim1, dims...)))
Random.randn(rng::RNG, T::NormalType, dim1::Integer, dims::Integer...; kwargs...) =
    Random.randn(rng, T, Dims((dim1, dims...)); kwargs...)
rand_logn(rng::RNG, T::LognormalType, dim1::Integer, dims::Integer...; kwargs...) =
    rand_logn(rng, T, Dims((dim1, dims...)); kwargs...)
rand_poisson(rng::RNG, T::PoissonType, dim1::Integer, dims::Integer...; kwargs...) =
    rand_poisson(rng, T, Dims((dim1, dims...)); kwargs...)

# scalars
Random.rand(rng::RNG, T::UniformType=Float32) = rand(rng, T, 1)[]
Random.randn(rng::RNG, T::NormalType=Float32; kwargs...) = randn(rng, T, 1; kwargs...)[]
rand_logn(rng::RNG, T::LognormalType=Float32; kwargs...) = rand_logn(rng, T, 1; kwargs...)[]
rand_poisson(rng::RNG, T::PoissonType=Float32; kwargs...) = rand_poisson(rng, T, 1; kwargs...)[]
