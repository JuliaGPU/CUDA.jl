# High-level rand/randn/seed! API and integration with CUDACore types

using CUDACore: AnyCuArray, CuArray, CuContext, active_state


## native RNG (stateless GPUArrays.RNG, no GPU resources to cache)

function native_rng()
    cuda = active_state()

    LibraryState = @NamedTuple{rng::GPUArrays.RNG}
    states = get!(task_local_storage(), :cuRAND_NativeRNG) do
        Dict{CuContext,LibraryState}()
    end::Dict{CuContext,LibraryState}

    @noinline function new_state(cuda)
        new_rng = GPUArrays.RNG()
        Random.seed!(new_rng)
        (; rng=new_rng)
    end
    state = get!(states, cuda.context) do
        new_state(cuda)
    end

    return state.rng
end


## cuRAND.rand / cuRAND.randn / cuRAND.seed! — high-level API

curand_rng() = default_rng()

function seed!(seed=Base.rand(UInt64))
    Random.seed!(native_rng(), seed)
    Random.seed!(curand_rng(), seed)
end

# CURAND in-place (convenience without explicit RNG)
Random.rand!(A::UniformArray) = Random.rand!(curand_rng(), A)
Random.randn!(A::NormalArray; kwargs...) = Random.randn!(curand_rng(), A; kwargs...)
rand_logn!(A::LognormalArray; kwargs...) = rand_logn!(curand_rng(), A; kwargs...)
rand_poisson!(A::PoissonArray; kwargs...) = rand_poisson!(curand_rng(), A; kwargs...)

# native in-place (fallback for types not supported by CURAND)
Random.rand!(A::AnyCuArray) = Random.rand!(native_rng(), A)
Random.randn!(A::AnyCuArray) = Random.randn!(native_rng(), A)
rand_logn!(A::AnyCuArray; kwargs...) =
    error("cuRAND does not support generating lognormally-distributed random numbers of type $(eltype(A))")
rand_poisson!(A::AnyCuArray; kwargs...) =
    error("cuRAND does not support generating Poisson-distributed random numbers of type $(eltype(A))")

# CURAND out-of-place
rand(T::UniformType, dims::Dims) = Random.rand(curand_rng(), T, dims)
randn(T::NormalType, dims::Dims; kwargs...) = Random.randn(curand_rng(), T, dims; kwargs...)
rand(T::UniformType, dim1::Integer, dims::Integer...) =
    Random.rand(curand_rng(), T, Dims((dim1, dims...)))
randn(T::NormalType, dim1::Integer, dims::Integer...; kwargs...) =
    Random.randn(curand_rng(), T, Dims((dim1, dims...)); kwargs...)

# native out-of-place
rand(T::Type, dims::Dims) = Random.rand!(CuArray{T}(undef, dims...))
randn(T::Type, dims::Dims; kwargs...) = Random.randn!(CuArray{T}(undef, dims...); kwargs...)
rand(T::Type, dim1::Integer, dims::Integer...) =
    Random.rand!(CuArray{T}(undef, dim1, dims...))
randn(T::Type, dim1::Integer, dims::Integer...; kwargs...) =
    Random.randn!(CuArray{T}(undef, dim1, dims...); kwargs...)

# untyped out-of-place (defaults to CURAND Float32)
rand(dim1::Integer, dims::Integer...) =
    Random.rand(curand_rng(), Dims((dim1, dims...)))
randn(dim1::Integer, dims::Integer...; kwargs...) =
    Random.randn(curand_rng(), Dims((dim1, dims...)); kwargs...)

# out-of-place logn/poisson
rand_logn(T::LognormalType, dims::Dims; kwargs...) = rand_logn(curand_rng(), T, dims; kwargs...)
rand_poisson(T::PoissonType, dims::Dims; kwargs...) = rand_poisson(curand_rng(), T, dims; kwargs...)
rand_logn(T::LognormalType, dim1::Integer, dims::Integer...; kwargs...) =
    rand_logn(curand_rng(), T, Dims((dim1, dims...)); kwargs...)
rand_poisson(T::PoissonType, dim1::Integer, dims::Integer...; kwargs...) =
    rand_poisson(curand_rng(), T, Dims((dim1, dims...)); kwargs...)
rand_logn(dim1::Integer, dims::Integer...; kwargs...) =
    rand_logn(curand_rng(), Dims((dim1, dims...)); kwargs...)
rand_poisson(dim1::Integer, dims::Integer...; kwargs...) =
    rand_poisson(curand_rng(), Dims((dim1, dims...)); kwargs...)

# scalars
rand(T::Type=Float32) = rand(T, 1)[]
randn(T::Type=Float32; kwargs...) = randn(T, 1; kwargs...)[]
rand_logn(T::Type=Float32; kwargs...) = rand_logn(curand_rng(), T, 1; kwargs...)[]
rand_poisson(T::Type=Cuint; kwargs...) = rand_poisson(curand_rng(), T, 1; kwargs...)[]
