# Native kernel-based RNG (doesn't use the cuRAND library)

using CUDACore: AnyCuArray, CuArray

"""
    cuRAND.NativeRNG()

A random number generator using `rand()` in a device kernel.

See also: `CUDACore.Philox2x32`
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

function advance_counter!(rng::NativeRNG)
    new_counter = Int64(rng.counter) + 1
    overflow, remainder = fldmod(new_counter, typemax(UInt32))
    rng.seed += overflow % UInt32
    rng.counter = remainder % UInt32
end


## Philox4x32-10 counter-based RNG
#
# Stateless: (counter, key) → 4 UInt32 outputs. Each unique counter gives independent
# random values with no shared memory or global state needed.
#
# Reference: Salmon et al., "Parallel Random Numbers: As Easy as 1, 2, 3" (2011)

const PHILOX_M4x32_0 = 0xD2511F53
const PHILOX_M4x32_1 = 0xCD9E8D57
const PHILOX_W32_0   = 0x9E3779B9
const PHILOX_W32_1   = 0xBB67AE85

@inline function philox4x32round(ctr::NTuple{4,UInt32}, key::NTuple{2,UInt32})
    mul0 = widemul(PHILOX_M4x32_0, ctr[1])
    mul1 = widemul(PHILOX_M4x32_1, ctr[3])
    hi0 = (mul0 >> 32) % UInt32
    hi1 = (mul1 >> 32) % UInt32
    lo0 = mul0 % UInt32
    lo1 = mul1 % UInt32
    (hi1 ⊻ ctr[2] ⊻ key[1], lo1, hi0 ⊻ ctr[4] ⊻ key[2], lo0)
end

@inline function philox4x32bumpkey(key::NTuple{2,UInt32})
    (key[1] + PHILOX_W32_0, key[2] + PHILOX_W32_1)
end

@inline function philox4x32_10(ctr::NTuple{4,UInt32}, key::NTuple{2,UInt32})
    ctr = philox4x32round(ctr, key); key = philox4x32bumpkey(key)
    ctr = philox4x32round(ctr, key); key = philox4x32bumpkey(key)
    ctr = philox4x32round(ctr, key); key = philox4x32bumpkey(key)
    ctr = philox4x32round(ctr, key); key = philox4x32bumpkey(key)
    ctr = philox4x32round(ctr, key); key = philox4x32bumpkey(key)
    ctr = philox4x32round(ctr, key); key = philox4x32bumpkey(key)
    ctr = philox4x32round(ctr, key); key = philox4x32bumpkey(key)
    ctr = philox4x32round(ctr, key); key = philox4x32bumpkey(key)
    ctr = philox4x32round(ctr, key); key = philox4x32bumpkey(key)
    ctr = philox4x32round(ctr, key)
    ctr
end


## Float conversions: unsigned integer → uniform float in (0, 1]

@inline function u01(::Type{Float32}, u::UInt32)
    fma(Float32(u), Float32(2)^(-32), Float32(2)^(-33))
end

# Pre-computed constants to avoid Float64 ^ calls that don't constant-fold on the GPU
const F64_2P_NEG64 = Float64(2)^(-64)
const F64_2P_NEG65 = Float64(2)^(-65)
const F64_2P_NEG66 = Float64(2)^(-66)

@inline function u01(::Type{Float64}, u::UInt64)
    fma(Float64(u), F64_2P_NEG64, F64_2P_NEG65)
end


## Fast log approximation for Box-Muller
#
# Polynomial approximation of log(u01(u)) adapted from fdlibm e_log.c.
# The u01 conversion is fused into the function to avoid precision loss.

const SQRT_HALF_I32 = reinterpret(Int32, Float32(sqrt(0.5)))
const LOG_ODD_F32   = (reinterpret(Float32, Int32(0x3f2aaaaa)),
                       reinterpret(Float32, Int32(0x3e91e9ee)))
const LOG_EVEN_F32  = (reinterpret(Float32, Int32(0x3eccce13)),
                       reinterpret(Float32, Int32(0x3e789e26)))

@inline function fast_log(::Type{Float32}, u::UInt32)
    x = fma(Float32(u), Float32(2)^(-32), Float32(2)^(-33))
    ix = reinterpret(Int32, x) - SQRT_HALF_I32
    k = ix >> Int32(23)
    f_std = reinterpret(Float32, (ix & Int32(0x007fffff)) + SQRT_HALF_I32) - 1.0f0
    f_comp = -fma(Float32(~u), Float32(2)^(-32), Float32(2)^(-33))
    f = ifelse(k == Int32(0), f_comp, f_std)
    s = f / (2.0f0 + f)
    z = s * s; w = z * z
    R = z * evalpoly(w, LOG_ODD_F32) + w * evalpoly(w, LOG_EVEN_F32)
    hfsq = 0.5f0 * f * f
    Float32(k) * reinterpret(Float32, Int32(0x3f317180)) -
        ((hfsq - (s * (hfsq + R) +
          Float32(k) * reinterpret(Float32, Int32(0x3717f7d1)))) - f)
end

const SQRT_HALF_I64 = reinterpret(Int64, sqrt(0.5))
const LOG_ODD_F64   = (6.666666666666735130e-01, 2.857142874366239149e-01,
                       1.818357216161805012e-01, 1.479819860511658591e-01)
const LOG_EVEN_F64  = (3.999999999940941908e-01, 2.222219843214978396e-01,
                       1.531383769920937332e-01)

@inline function fast_log(::Type{Float64}, u::UInt64)
    x = fma(Float64(u), F64_2P_NEG64, F64_2P_NEG65)
    ix = reinterpret(Int64, x) - SQRT_HALF_I64
    k = ix >> Int64(52)
    f_std = reinterpret(Float64, (ix & Int64(0x000fffffffffffff)) + SQRT_HALF_I64) - 1.0
    f_comp = -fma(Float64(~u), F64_2P_NEG64, F64_2P_NEG65)
    f = ifelse(k == Int64(0), f_comp, f_std)
    s = f / (2.0 + f)
    z = s * s; w = z * z
    R = z * evalpoly(w, LOG_ODD_F64) + w * evalpoly(w, LOG_EVEN_F64)
    hfsq = 0.5 * f * f
    Float64(k) * 6.93147180369123816490e-01 -
        ((hfsq - (s * (hfsq + R) + Float64(k) * 1.90821492927058500170e-10)) - f)
end


## Fast sincospi for Box-Muller
#
# Minimax polynomial approximation of (sin(θ), cos(θ)) at a uniformly random angle.
# Bottom 3 bits of u select one of 8 octants; upper bits give the reduced argument.
# A +0.5 bias avoids y=0. Branchless octant handling via bit ops.

const SINPI_F32 = (3.1415927f0, -5.167708f0, 2.5497673f0, -0.58907866f0)
const COSPI_F32 = (1.0f0, -4.934788f0, 4.057578f0, -1.3061346f0)

@inline function fast_sincospi(::Type{Float32}, u::UInt32)
    oct = (u % Int32) & Int32(7)
    y = fma(Float32(u & ~UInt32(7)), Float32(2)^(-34), Float32(2)^(-32))
    sp = y * evalpoly(y * y, SINPI_F32)
    cp = evalpoly(y * y, COSPI_F32)
    swap    = !iszero(oct & Int32(1))
    sin_neg = !iszero(oct & Int32(2))
    cos_neg = !iszero(oct & Int32(4))
    s_raw = ifelse(swap, cp, sp)
    c_raw = ifelse(swap, sp, cp)
    (ifelse(sin_neg, -s_raw, s_raw), ifelse(cos_neg, -c_raw, c_raw))
end

const SINPI_F64 = (3.141592653589793, -5.167712780049954, 2.5501640398733785,
                   -0.5992645289398095, 0.08214586918507949, -0.007370021659123395,
                   0.0004615322405282014)
const COSPI_F64 = (1.0, -4.934802200544605, 4.0587121263978485,
                   -1.3352627670374702, 0.23533054723811608, -0.025804938901032953,
                   0.0019068114005246046)

@inline function fast_sincospi(::Type{Float64}, u::UInt64)
    oct = (u % Int32) & Int32(7)
    y = fma(Float64(u & ~UInt64(7)), F64_2P_NEG66, F64_2P_NEG64)
    sp = y * evalpoly(y * y, SINPI_F64)
    cp = evalpoly(y * y, COSPI_F64)
    swap    = !iszero(oct & Int32(1))
    sin_neg = !iszero(oct & Int32(2))
    cos_neg = !iszero(oct & Int32(4))
    s_raw = ifelse(swap, cp, sp)
    c_raw = ifelse(swap, sp, cp)
    (ifelse(sin_neg, -s_raw, s_raw), ifelse(cos_neg, -c_raw, c_raw))
end


## Box-Muller transform: 2 uniform UInts → 2 normal floats

@inline function boxmuller(::Type{Float32}, u1::UInt32, u2::UInt32)
    r = Base.sqrt_llvm(-2 * fast_log(Float32, u2))
    s, c = fast_sincospi(Float32, u1)
    (r * s, r * c)
end

@inline function boxmuller(::Type{Float64}, u1::UInt64, u2::UInt64)
    r = Base.sqrt_llvm(-2 * fast_log(Float64, u2))
    s, c = fast_sincospi(Float64, u1)
    (r * s, r * c)
end


## rand!

const RNG_THREADS = 256

function Random.rand!(rng::NativeRNG, A::AnyCuArray{Float32})
    isempty(A) && return A

    function kernel(A::AbstractArray{Float32}, seed::UInt32, counter::UInt32)
        tid = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x
        stride = gridDim().x * blockDim().x
        len = length(A)
        i = tid
        while 4 * i <= len
            a1, a2, a3, a4 = philox4x32_10(
                (i % UInt32, UInt32(0), counter, UInt32(0)),
                (seed, UInt32(0)))
            @inbounds A[4*i - 3] = u01(Float32, a1)
            @inbounds A[4*i - 2] = u01(Float32, a2)
            @inbounds A[4*i - 1] = u01(Float32, a3)
            @inbounds A[4*i]     = u01(Float32, a4)
            i += stride
        end
        if threadIdx().x == 1i32 && blockIdx().x == 1i32
            rem = len % 4
            if rem > 0
                base = len - rem
                idx = (base ÷ 4 + 1) % UInt32
                a1, a2, a3, a4 = philox4x32_10(
                    (idx, UInt32(0), counter, UInt32(0)),
                    (seed, UInt32(0)))
                vals = (u01(Float32, a1), u01(Float32, a2),
                        u01(Float32, a3), u01(Float32, a4))
                for j in 1:rem
                    @inbounds A[base + j] = vals[j]
                end
            end
        end
        return
    end

    blocks = cld(cld(length(A), 4), RNG_THREADS)
    @cuda threads=RNG_THREADS blocks=blocks name="rand!" kernel(A, rng.seed, rng.counter)
    advance_counter!(rng)
    A
end

function Random.rand!(rng::NativeRNG, A::AnyCuArray{Float64})
    isempty(A) && return A

    function kernel(A::AbstractArray{Float64}, seed::UInt32, counter::UInt32)
        tid = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x
        stride = gridDim().x * blockDim().x
        len = length(A)
        i = tid
        while 2 * i <= len
            a1, a2, a3, a4 = philox4x32_10(
                (i % UInt32, UInt32(0), counter, UInt32(0)),
                (seed, UInt32(0)))
            @inbounds A[2*i - 1] = u01(Float64, UInt64(a1) | UInt64(a2) << 32)
            @inbounds A[2*i]     = u01(Float64, UInt64(a3) | UInt64(a4) << 32)
            i += stride
        end
        if threadIdx().x == 1i32 && blockIdx().x == 1i32 && isodd(len)
            idx = (len ÷ 2 + 1) % UInt32
            a1, a2, _, _ = philox4x32_10(
                (idx, UInt32(0), counter, UInt32(0)),
                (seed, UInt32(0)))
            @inbounds A[len] = u01(Float64, UInt64(a1) | UInt64(a2) << 32)
        end
        return
    end

    blocks = cld(cld(length(A), 2), RNG_THREADS)
    @cuda threads=RNG_THREADS blocks=blocks name="rand!" kernel(A, rng.seed, rng.counter)
    advance_counter!(rng)
    A
end

# Generic fallback for types without optimized kernels
function Random.rand!(rng::NativeRNG, A::AnyCuArray)
    isempty(A) && return A

    function kernel(A::AbstractArray{T}, seed::UInt32, counter::UInt32) where {T}
        device_rng = Random.default_rng()
        @inbounds Random.seed!(device_rng, seed, counter)

        threadId = threadIdx().x
        window = Int64(blockDim().x) * Int64(gridDim().x)
        offset = Int64(blockIdx().x - 1i32) * Int64(blockDim().x)
        while offset < length(A)
            i = threadId + offset
            if i <= length(A)
                @inbounds A[i] = Random.rand(device_rng, T)
            end
            offset += window
        end
        return
    end

    blocks = cld(length(A), RNG_THREADS)
    @cuda threads=RNG_THREADS blocks=blocks name="rand!" kernel(A, rng.seed, rng.counter)

    new_counter = Int64(rng.counter) + length(A)
    overflow, remainder = fldmod(new_counter, typemax(UInt32))
    rng.seed += overflow
    rng.counter = remainder
    A
end


## randn!

function Random.randn!(rng::NativeRNG, A::AnyCuArray{Float32})
    isempty(A) && return A

    function kernel(A::AbstractArray{Float32}, seed::UInt32, counter::UInt32)
        tid = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x
        stride = gridDim().x * blockDim().x
        len = length(A)
        i = tid
        while 4 * i <= len
            a1, a2, a3, a4 = philox4x32_10(
                (i % UInt32, UInt32(0), counter, UInt32(0)),
                (seed, UInt32(0)))
            n1, n2 = boxmuller(Float32, a1, a2)
            n3, n4 = boxmuller(Float32, a3, a4)
            @inbounds A[4*i - 3] = n1
            @inbounds A[4*i - 2] = n2
            @inbounds A[4*i - 1] = n3
            @inbounds A[4*i]     = n4
            i += stride
        end
        if threadIdx().x == 1i32 && blockIdx().x == 1i32
            rem = len % 4
            if rem > 0
                base = len - rem
                idx = (base ÷ 4 + 1) % UInt32
                a1, a2, a3, a4 = philox4x32_10(
                    (idx, UInt32(0), counter, UInt32(0)),
                    (seed, UInt32(0)))
                n1, n2 = boxmuller(Float32, a1, a2)
                n3, n4 = boxmuller(Float32, a3, a4)
                vals = (n1, n2, n3, n4)
                for j in 1:rem
                    @inbounds A[base + j] = vals[j]
                end
            end
        end
        return
    end

    blocks = cld(cld(length(A), 4), RNG_THREADS)
    @cuda threads=RNG_THREADS blocks=blocks name="randn!" kernel(A, rng.seed, rng.counter)
    advance_counter!(rng)
    A
end

function Random.randn!(rng::NativeRNG, A::AnyCuArray{Float64})
    isempty(A) && return A

    function kernel(A::AbstractArray{Float64}, seed::UInt32, counter::UInt32)
        tid = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x
        stride = gridDim().x * blockDim().x
        len = length(A)
        i = tid
        while 2 * i <= len
            a1, a2, a3, a4 = philox4x32_10(
                (i % UInt32, UInt32(0), counter, UInt32(0)),
                (seed, UInt32(0)))
            n1, n2 = boxmuller(Float64,
                UInt64(a1) | UInt64(a2) << 32,
                UInt64(a3) | UInt64(a4) << 32)
            @inbounds A[2*i - 1] = n1
            @inbounds A[2*i]     = n2
            i += stride
        end
        if threadIdx().x == 1i32 && blockIdx().x == 1i32 && isodd(len)
            idx = (len ÷ 2 + 1) % UInt32
            a1, a2, a3, a4 = philox4x32_10(
                (idx, UInt32(0), counter, UInt32(0)),
                (seed, UInt32(0)))
            n1, _ = boxmuller(Float64,
                UInt64(a1) | UInt64(a2) << 32,
                UInt64(a3) | UInt64(a4) << 32)
            @inbounds A[len] = n1
        end
        return
    end

    blocks = cld(cld(length(A), 2), RNG_THREADS)
    @cuda threads=RNG_THREADS blocks=blocks name="randn!" kernel(A, rng.seed, rng.counter)
    advance_counter!(rng)
    A
end

function Random.randn!(rng::NativeRNG, A::AnyCuArray{Complex{Float32}})
    isempty(A) && return A

    function kernel(A::AbstractArray{Complex{Float32}}, seed::UInt32, counter::UInt32)
        tid = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x
        stride = gridDim().x * blockDim().x
        len = length(A)
        i = tid
        # 4 UInt32 outputs → 2 complex values (each needs 2 normals)
        while 2 * i <= len
            a1, a2, a3, a4 = philox4x32_10(
                (i % UInt32, UInt32(0), counter, UInt32(0)),
                (seed, UInt32(0)))
            # sqrt(-log(U)) not sqrt(-2*log(U)): each component has variance 1/2
            r1 = Base.sqrt_llvm(-fast_log(Float32, a2))
            s1, c1 = fast_sincospi(Float32, a1)
            r2 = Base.sqrt_llvm(-fast_log(Float32, a4))
            s2, c2 = fast_sincospi(Float32, a3)
            @inbounds A[2*i - 1] = complex(r1 * s1, r1 * c1)
            @inbounds A[2*i]     = complex(r2 * s2, r2 * c2)
            i += stride
        end
        if threadIdx().x == 1i32 && blockIdx().x == 1i32 && isodd(len)
            idx = (len ÷ 2 + 1) % UInt32
            a1, a2, _, _ = philox4x32_10(
                (idx, UInt32(0), counter, UInt32(0)),
                (seed, UInt32(0)))
            r = Base.sqrt_llvm(-fast_log(Float32, a2))
            s, c = fast_sincospi(Float32, a1)
            @inbounds A[len] = complex(r * s, r * c)
        end
        return
    end

    blocks = cld(cld(length(A), 2), RNG_THREADS)
    @cuda threads=RNG_THREADS blocks=blocks name="randn!" kernel(A, rng.seed, rng.counter)
    advance_counter!(rng)
    A
end

function Random.randn!(rng::NativeRNG, A::AnyCuArray{Complex{Float64}})
    isempty(A) && return A

    function kernel(A::AbstractArray{Complex{Float64}}, seed::UInt32, counter::UInt32)
        tid = (blockIdx().x - 1i32) * blockDim().x + threadIdx().x
        stride = gridDim().x * blockDim().x
        len = length(A)
        i = tid
        # 4 UInt32 → 2 UInt64 → 1 complex value per Philox call
        while i <= len
            a1, a2, a3, a4 = philox4x32_10(
                (i % UInt32, UInt32(0), counter, UInt32(0)),
                (seed, UInt32(0)))
            u1 = UInt64(a1) | UInt64(a2) << 32
            u2 = UInt64(a3) | UInt64(a4) << 32
            r = Base.sqrt_llvm(-fast_log(Float64, u2))
            s, c = fast_sincospi(Float64, u1)
            @inbounds A[i] = complex(r * s, r * c)
            i += stride
        end
        return
    end

    blocks = cld(length(A), RNG_THREADS)
    @cuda threads=RNG_THREADS blocks=blocks name="randn!" kernel(A, rng.seed, rng.counter)
    advance_counter!(rng)
    A
end

# Generic randn! fallback for other float/complex types
function Random.randn!(rng::NativeRNG, A::AnyCuArray{<:Union{AbstractFloat,Complex{<:AbstractFloat}}})
    isempty(A) && return A

    function kernel(A::AbstractArray{T}, seed::UInt32, counter::UInt32) where {T<:Real}
        device_rng = Random.default_rng()
        @inbounds Random.seed!(device_rng, seed, counter)

        threadId = threadIdx().x
        window = Int64(blockDim().x) * Int64(gridDim().x)
        offset = Int64(blockIdx().x - 1i32) * Int64(blockDim().x)
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
        window = Int64(blockDim().x) * Int64(gridDim().x)
        offset = Int64(blockIdx().x - 1i32) * Int64(blockDim().x)
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

    blocks = cld(cld(length(A), 2), RNG_THREADS)
    @cuda threads=RNG_THREADS blocks=blocks name="randn!" kernel(A, rng.seed, rng.counter)

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
