# High-level rand/randn/seed! API and integration with CUDACore types

using CUDACore: AnyCuArray, CuArray, CuContext, active_state
using CUDACore: GPUArrays


## native RNG handle cache (kernel-based Philox2x32)

function native_rng_ctor(ctx)
    context!(ctx) do
        NativeRNG()
    end
end
function native_rng_dtor(ctx, rng) end
const idle_native_rngs = HandleCache{CuContext,NativeRNG}(native_rng_ctor, native_rng_dtor)

function native_rng()
    cuda = active_state()

    LibraryState = @NamedTuple{rng::NativeRNG}
    states = get!(task_local_storage(), :cuRAND_NativeRNG) do
        Dict{CuContext,LibraryState}()
    end::Dict{CuContext,LibraryState}

    @noinline function new_state(cuda)
        new_rng = pop!(idle_native_rngs, cuda.context)
        finalizer(current_task()) do task
            push!(idle_native_rngs, cuda.context, new_rng)
        end
        (; rng=new_rng)
    end
    state = get!(states, cuda.context) do
        new_state(cuda)
    end

    return state.rng
end


## cuRAND.rand / cuRAND.randn / cuRAND.seed! — high-level API

# The implicit default for CuArray dispatch is the GPUArrays RNG (stateless
# Philox4x32-10 batched kernels). NativeRNG and LibraryRNG are available
# explicitly via native_rng() and library_rng() for perf comparison / RNGTest.
function seed!(seed=Base.rand(UInt64))
    Random.seed!(GPUArrays.default_rng(CuArray), seed)
    Random.seed!(native_rng(), seed)
    Random.seed!(library_rng(), seed)
end

# cuRAND in-place (convenience without explicit RNG, for the types cuRAND supports)
Random.rand!(A::UniformArray) = Random.rand!(library_rng(), A)
Random.randn!(A::NormalArray; kwargs...) = Random.randn!(library_rng(), A; kwargs...)
rand_logn!(A::LognormalArray; kwargs...) = rand_logn!(library_rng(), A; kwargs...)
rand_poisson!(A::PoissonArray; kwargs...) = rand_poisson!(library_rng(), A; kwargs...)

# GPUArrays RNG fallback for types not supported by cuRAND
Random.rand!(A::AnyCuArray) = Random.rand!(GPUArrays.default_rng(CuArray), A)
Random.randn!(A::AnyCuArray) = Random.randn!(GPUArrays.default_rng(CuArray), A)
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
