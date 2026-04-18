# High-level rand/randn/seed! API and integration with CUDACore types

using CUDACore: AnyCuArray, CuArray, CuContext, active_state
using CUDACore: GPUArrays


## native RNG (kernel-based Philox2x32)
#
# Holds no GPU memory, just two UInt32s, so no HandleCache is needed: each
# task constructs its own and TLS-drop / task-GC frees it.

const native_state_cache = CUDACore.TaskLocalCache{CuContext, NativeRNG}(:cuRAND_NativeRNG)

function native_rng()
    cuda = active_state()
    states = CUDACore.task_dict(native_state_cache)
    get!(() -> NativeRNG(), states, cuda.context)
end


## default RNG: task-local cached GPUArrays.RNG{CuArray} (the fast Philox4x32-10
## batched-kernel RNG). Used by Random.rand!/randn!(::AnyCuArray) when no rng
## is supplied, and exposed via CUDA.RNG / CUDA.gpuarrays_rng().

const gpuarrays_state_cache = CUDACore.TaskLocalCache{CuContext, GPUArrays.RNG{CuArray}}(:cuRAND_DefaultRNG)

function gpuarrays_rng()
    cuda = active_state()
    states = CUDACore.task_dict(gpuarrays_state_cache)
    get!(states, cuda.context) do
        new_rng = GPUArrays.RNG{CuArray}()
        Random.seed!(new_rng)
        new_rng
    end
end


## cuRAND.rand / cuRAND.randn / cuRAND.seed! — high-level API

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
Random.randn!(A::AnyCuArray) = Random.randn!(gpuarrays_rng(), A)
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
