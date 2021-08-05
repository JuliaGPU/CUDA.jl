# random functions that dispatch either to CURAND or GPUArrays' generic RNG

using Random

export rand_logn!, rand_poisson!


# native RNG

"""
    CUDA.RNG()

A random number generator using `rand()` in a device kernel.

See also: [CUDA.Philox2x32](@ref)
"""
mutable struct RNG <: AbstractRNG
    seed::UInt32
    counter::UInt32

    function RNG(seed::Integer)
        new(seed%UInt32, 0)
    end
end

RNG() = RNG(Random.rand(UInt32))

function Random.seed!(rng::RNG, seed::Integer)
    rng.seed = seed % UInt32
    rng.counter = 0
end

Random.seed!(rng::RNG) = seed!(rng, Random.rand(UInt32))

function Random.rand!(rng::RNG, A::AnyCuArray)
    function kernel(A::AbstractArray{T}, seed::UInt32, counter::UInt32) where {T}
        device_rng = Random.default_rng()

        # initialize the state
        @inbounds Random.seed!(device_rng, seed, counter)

        # grid-stride loop
        threadId = threadIdx().x
        window = (blockDim().x - 1) * gridDim().x
        offset = (blockIdx().x - 1) * blockDim().x
        while offset < length(A)
            i = threadId + offset
            if i <= length(A)
                @inbounds A[i] = Random.rand(device_rng, T)
            end

            offset += window
        end

        return
    end

    kernel = @cuda launch=false name="rand!" kernel(A, rng.seed, rng.counter)
    config = launch_configuration(kernel.fun; max_threads=64)
    threads = max(32, min(config.threads, length(A)))
    blocks = min(config.blocks, cld(length(A), threads))
    kernel(A, rng.seed, rng.counter; threads=threads, blocks=blocks)

    new_counter = Int64(rng.counter) + length(A)
    overflow, remainder = fldmod(new_counter, typemax(UInt32))
    rng.seed += overflow     # XXX: is this OK?
    rng.counter = remainder

    A
end

function Random.randn!(rng::RNG, A::AnyCuArray{<:T}) where {T<:Union{AbstractFloat,Complex{<:AbstractFloat}}}
    function kernel(A::AbstractArray{T}, seed::UInt32, counter::UInt32)
        device_rng = Random.default_rng()

        # initialize the state
        @inbounds Random.seed!(device_rng, seed, counter)

        # grid-stride loop
        threadId = threadIdx().x
        window = (blockDim().x - 1) * gridDim().x
        offset = (blockIdx().x - 1) * blockDim().x
        while offset < length(A)
            i = threadId + offset
            j = threadId + offset + window
            if i <= length(A)
                # Box–Muller transform
                U1 = Random.rand(device_rng, T)
                U2 = Random.rand(device_rng, T)
                Z0 = sqrt(T(-2.0)*log(U1))*cos(T(2pi)*U2)
                Z1 = sqrt(T(-2.0)*log(U1))*sin(T(2pi)*U2)
                @inbounds A[i] = Z0
                if j <= length(A)
                    @inbounds A[j] = Z1
                end
            end

            offset += 2*window
        end

        return
    end

    kernel = @cuda launch=false name="rand!" kernel(A, rng.seed, rng.counter)
    config = launch_configuration(kernel.fun; max_threads=64)
    threads = max(32, min(config.threads, length(A)÷2))
    blocks = min(config.blocks, cld(length(A)÷2, threads))
    kernel(A, rng.seed, rng.counter; threads=threads, blocks=blocks)

    new_counter = Int64(rng.counter) + length(A)
    overflow, remainder = fldmod(new_counter, typemax(UInt32))
    rng.seed += overflow     # XXX: is this OK?
    rng.counter = remainder

    A
end


# generic functionality

function Random.rand!(rng::Union{RNG,CURAND.RNG,GPUArrays.RNG}, A::AbstractArray{T}) where {T}
    B = CuArray{T}(undef, size(A))
    Random.rand!(rng, B)
    copyto!(A, B)
end

function Random.rand(rng::Union{RNG,CURAND.RNG,GPUArrays.RNG}, T::Type)
    assertscalar("scalar rand")
    A = CuArray{T}(undef, 1)
    Random.rand!(rng, A)
    A[]
end


# RNG-less interface

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
