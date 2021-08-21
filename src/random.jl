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

function default_rng()
    cuda = CUDA.active_state()

    # every task maintains library state per device
    LibraryState = @NamedTuple{rng::RNG}
    states = get!(task_local_storage(), :RNG) do
        Dict{CuContext,LibraryState}()
    end::Dict{CuContext,LibraryState}

    # get library state
    @noinline function new_state(cuda)
        # CUDA RNG objects are cheap, so we don't need to cache them
        (; rng=RNG())
    end
    state = get!(states, cuda.context) do
        new_state(cuda)
    end

    return state.rng
end


# old native RNG

# we keep this for the GPUArrays.jl tests

const idle_gpuarray_rngs = HandleCache{CuContext,GPUArrays.RNG}()

function GPUArrays.default_rng(::Type{<:CuArray})
    cuda = CUDA.active_state()

    # every task maintains library state per device
    LibraryState = @NamedTuple{rng::GPUArrays.RNG}
    states = get!(task_local_storage(), :GPUArraysRNG) do
        Dict{CuContext,LibraryState}()
    end::Dict{CuContext,LibraryState}

    # get library state
    @noinline function new_state(cuda)
        new_rng = pop!(idle_gpuarray_rngs, cuda.context) do
            N = attribute(cuda.device, DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
            buf = CuArray{NTuple{4, UInt32}}(undef, N)
            GPUArrays.RNG(buf)
        end

        finalizer(current_task()) do task
            push!(idle_gpuarray_rngs, cuda.context, new_rng) do
                # no need to do anything, as the RNG is collected by its finalizer
            end
        end

        Random.seed!(new_rng)
        (; rng=new_rng)
    end
    state = get!(states, cuda.context) do
        new_state(cuda)
    end

    return state.rng
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

cuda_rng() = default_rng()
curand_rng() = CURAND.default_rng()

function seed!(seed=Base.rand(UInt64))
    Random.seed!(cuda_rng(), seed)
    Random.seed!(curand_rng(), seed)
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

# native in-place
Random.rand!(A::AnyCuArray) = Random.rand!(cuda_rng(), A)
Random.randn!(A::AnyCuArray) = Random.randn!(cuda_rng(), A)
rand_logn!(A::AnyCuArray; kwargs...) =
    error("CUDA.jl does not support generating lognormally-distributed random numbers of type $(eltype(A))")
rand_poisson!(A::AnyCuArray; kwargs...) =
    error("CUDA.jl does not support generating Poisson-distributed random numbers of type $(eltype(A))")

# native out-of-place
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
