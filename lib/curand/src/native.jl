# Native kernel-based RNG (doesn't use the cuRAND library)

using CUDA: AnyCuArray, CuArray

"""
    cuRAND.NativeRNG()

A random number generator using `rand()` in a device kernel.

See also: [CUDA.Philox2x32](@ref)
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

function Random.rand!(rng::NativeRNG, A::AnyCuArray)
    isempty(A) && return A

    function kernel(A::AbstractArray{T}, seed::UInt32, counter::UInt32) where {T}
        device_rng = Random.default_rng()
        @inbounds Random.seed!(device_rng, seed, counter)

        threadId = threadIdx().x
        window = widemul(blockDim().x, gridDim().x)
        offset = widemul(blockIdx().x - 1i32, blockDim().x)
        while offset < length(A)
            i = threadId + offset
            if i <= length(A)
                @inbounds A[i] = Random.rand(device_rng, T)
            end
            offset += window
        end
        return
    end

    threads = 32
    blocks = cld(length(A), threads)
    @cuda threads=threads blocks=blocks name="rand!" kernel(A, rng.seed, rng.counter)

    new_counter = Int64(rng.counter) + length(A)
    overflow, remainder = fldmod(new_counter, typemax(UInt32))
    rng.seed += overflow
    rng.counter = remainder
    A
end

function Random.randn!(rng::NativeRNG, A::AnyCuArray{<:Union{AbstractFloat,Complex{<:AbstractFloat}}})
    isempty(A) && return A

    function kernel(A::AbstractArray{T}, seed::UInt32, counter::UInt32) where {T<:Real}
        device_rng = Random.default_rng()
        @inbounds Random.seed!(device_rng, seed, counter)

        threadId = threadIdx().x
        window = widemul(blockDim().x, gridDim().x)
        offset = widemul(blockIdx().x - 1i32, blockDim().x)
        while offset < length(A)
            i = threadId + offset
            j = threadId + offset + window
            if i <= length(A)
                U1 = Random.rand(device_rng, T)
                while U1 == zero(T)
                    U1 = Random.rand(device_rng, T)
                end
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

    function kernel(A::AbstractArray{Complex{T}}, seed::UInt32, counter::UInt32) where {T<:Real}
        device_rng = Random.default_rng()
        @inbounds Random.seed!(device_rng, seed, counter)

        threadId = threadIdx().x
        window = widemul(blockDim().x, gridDim().x)
        offset = widemul(blockIdx().x - 1i32, blockDim().x)
        while offset < length(A)
            i = threadId + offset
            if i <= length(A)
                U1 = Random.rand(device_rng, T)
                while U1 == zero(T)
                    U1 = Random.rand(device_rng, T)
                end
                U2 = Random.rand(device_rng, T)
                Z0 = sqrt(-log(U1))*cos(T(2pi)*U2)
                Z1 = sqrt(-log(U1))*sin(T(2pi)*U2)
                @inbounds A[i] = complex(Z0, Z1)
            end
            offset += window
        end
        return
    end

    threads = 32
    blocks = cld(cld(length(A), 2), threads)
    @cuda threads=threads blocks=blocks name="randn!" kernel(A, rng.seed, rng.counter)

    new_counter = Int64(rng.counter) + length(A)
    overflow, remainder = fldmod(new_counter, typemax(UInt32))
    rng.seed += overflow
    rng.counter = remainder
    A
end

# NativeRNG out-of-place
Random.rand(rng::NativeRNG, T::Type, dims::Dims) =
    rand!(rng, CuArray{T}(undef, dims))
Random.randn(rng::NativeRNG, T::Type, dims::Dims) =
    randn!(rng, CuArray{T}(undef, dims))
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
    rand!(rng, B)
    copyto!(A, B)
end
function Random.randn!(rng::NativeRNG, A::AbstractArray{T}) where {T}
    B = CuArray{T}(undef, size(A))
    randn!(rng, B)
    copyto!(A, B)
end

# scalars
Random.rand(rng::NativeRNG, T::Type=Float32) = Random.rand(rng, T, 1)[]
Random.randn(rng::NativeRNG, T::Type=Float32) = Random.randn(rng, T, 1)[]
Random.randn(rng::NativeRNG, T::Random.BitFloatType) = Random.randn(rng, T, 1)[]
