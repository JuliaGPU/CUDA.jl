# Native kernel-based RNG (doesn't use the cuRAND library)
#
# Builds on the device-side Philox2x32 RNG (CUDACore.Philox2x32). Kept separate
# from the GPUArrays-based RNG to allow perf comparison, RNGTest runs, etc.

using CUDACore: AnyCuArray, CuArray

"""
    cuRAND.NativeRNG()

A random number generator that launches a CUDA kernel which calls the
device-side `rand()`/`randn()` on CUDACore's Philox2x32 generator.

!!! warning
    This RNG exists for testing, RNGTest statistical validation, and perf
    comparison against alternative designs (cuRAND library, GPUArrays.RNG,
    PhiloxRNG.jl). For production use prefer [`CUDA.RNG`](@ref), which
    wraps the batched-kernel Philox4x32-10 RNG from GPUArrays and is several
    times faster for bulk `rand!`/`randn!` calls.

See also: `CUDACore.Philox2x32`, `CUDA.RNG`.
"""
mutable struct NativeRNG <: Random.AbstractRNG
    seed::UInt32
    counter::UInt32

    function NativeRNG(seed::Integer)
        new(seed%UInt32, 0)
    end
    NativeRNG(seed::UInt32, counter::UInt32) = new(seed, counter)
end

native_make_seed() = Base.rand(Random.RandomDevice(), UInt32)

NativeRNG() = NativeRNG(native_make_seed())

Base.copy(rng::NativeRNG) = NativeRNG(rng.seed, rng.counter)
Base.hash(rng::NativeRNG, h::UInt) = hash(rng.seed, hash(rng.counter, h))
Base.:(==)(a::NativeRNG, b::NativeRNG) = (a.seed == b.seed) && (a.counter == b.counter)

function Random.seed!(rng::NativeRNG, seed::Integer)
    rng.seed = seed % UInt32
    rng.counter = 0
end

Random.seed!(rng::NativeRNG) = Random.seed!(rng, native_make_seed())

# Grid-stride kernel that fills A by calling `f(device_rng, T)` for each slot.
@inline function _native_fill!(f::F, A::AbstractArray{T}, seed::UInt32,
                                counter::UInt32) where {F, T}
    device_rng = Random.default_rng()
    @inbounds Random.seed!(device_rng, seed, counter)
    threadId = threadIdx().x
    window = widemul(blockDim().x, gridDim().x)
    offset = widemul(blockIdx().x - 1i32, blockDim().x)
    while offset < length(A)
        i = threadId + offset
        if i <= length(A)
            @inbounds A[i] = f(device_rng, T)
        end
        offset += window
    end
    return
end

function _native_advance!(rng::NativeRNG, n::Integer)
    new_counter = Int64(rng.counter) + n
    overflow, remainder = fldmod(new_counter, typemax(UInt32))
    rng.seed += overflow
    rng.counter = remainder
    return
end

function Random.rand!(rng::NativeRNG, A::AnyCuArray)
    isempty(A) && return A
    function kernel(A, seed, counter)
        _native_fill!(Random.rand, A, seed, counter)
    end
    threads = 32
    blocks = cld(length(A), threads)
    @cuda threads=threads blocks=blocks name="rand!" kernel(A, rng.seed, rng.counter)
    _native_advance!(rng, length(A))
    A
end

function Random.randn!(rng::NativeRNG, A::AnyCuArray{<:Union{AbstractFloat,
                                                              Complex{<:AbstractFloat}}})
    isempty(A) && return A
    function kernel(A, seed, counter)
        _native_fill!(Random.randn, A, seed, counter)
    end
    threads = 32
    blocks = cld(length(A), threads)
    @cuda threads=threads blocks=blocks name="randn!" kernel(A, rng.seed, rng.counter)
    _native_advance!(rng, length(A))
    A
end

# out-of-place
Random.rand(rng::NativeRNG, T::Type, dims::Dims) =
    Random.rand!(rng, CuArray{T}(undef, dims))
Random.randn(rng::NativeRNG, T::Type, dims::Dims) =
    Random.randn!(rng, CuArray{T}(undef, dims))
Random.rand(rng::NativeRNG, dims::Dims) = Random.rand(rng, Float32, dims)
Random.randn(rng::NativeRNG, dims::Dims) = Random.randn(rng, Float32, dims)
Random.rand(rng::NativeRNG, dim1::Integer, dims::Integer...) =
    Random.rand(rng, Dims((dim1, dims...)))
Random.randn(rng::NativeRNG, dim1::Integer, dims::Integer...) =
    Random.randn(rng, Dims((dim1, dims...)))
Random.rand(rng::NativeRNG, T::Type, dim1::Integer, dims::Integer...) =
    Random.rand(rng, T, Dims((dim1, dims...)))
Random.randn(rng::NativeRNG, T::Type, dim1::Integer, dims::Integer...) =
    Random.randn(rng, T, Dims((dim1, dims...)))

# CPU array fallback
function Random.rand!(rng::NativeRNG, A::AbstractArray{T}) where {T}
    B = CuArray{T}(undef, size(A))
    Random.rand!(rng, B)
    copyto!(A, B)
end
function Random.randn!(rng::NativeRNG, A::AbstractArray{T}) where {T}
    B = CuArray{T}(undef, size(A))
    Random.randn!(rng, B)
    copyto!(A, B)
end

# scalars
Random.rand(rng::NativeRNG, T::Type=Float32) = Random.rand(rng, T, 1)[]
Random.randn(rng::NativeRNG, T::Type=Float32) = Random.randn(rng, T, 1)[]
Random.randn(rng::NativeRNG, T::Random.BitFloatType) = Random.randn(rng, T, 1)[]
