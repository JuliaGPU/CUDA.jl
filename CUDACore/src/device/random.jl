## random number generation

using Random
import RandomNumbers


# global state

# we cannot store RNG state in thread-local memory (i.e. in the `rng` object) because that
# inflate register usage. instead, we store it in shared memory, with one entry per warp.
#
# XXX: this implies that state is shared between `rng` objects, which can be surprising.

# array with seeds, per warp, initialized on kernel start or by calling `seed!`
@eval @inline function global_random_keys()
    ptr = Base.llvmcall(
        $("""@global_random_keys = weak addrspace($(AS.Shared)) global [32 x i32] zeroinitializer, align 32
             define i8 addrspace($(AS.Shared))* @entry() #0 {
                 %ptr = getelementptr inbounds [32 x i32], [32 x i32] addrspace($(AS.Shared))* @global_random_keys, i64 0, i64 0
                 %untyped_ptr = bitcast i32 addrspace($(AS.Shared))* %ptr to i8 addrspace($(AS.Shared))*
                 ret i8 addrspace($(AS.Shared))* %untyped_ptr
             }
             attributes #0 = { alwaysinline }
          """, "entry"), LLVMPtr{UInt32, AS.Shared}, Tuple{})
    CuDeviceArray{UInt32,1,AS.Shared}(ptr, (32,))
end

# array with per-warp counters, incremented when generating numbers
@eval @inline function global_random_counters()
    ptr = Base.llvmcall(
        $("""@global_random_counters = weak addrspace($(AS.Shared)) global [32 x i32] zeroinitializer, align 32
             define i8 addrspace($(AS.Shared))* @entry() #0 {
                 %ptr = getelementptr inbounds [32 x i32], [32 x i32] addrspace($(AS.Shared))* @global_random_counters, i64 0, i64 0
                 %untyped_ptr = bitcast i32 addrspace($(AS.Shared))* %ptr to i8 addrspace($(AS.Shared))*
                 ret i8 addrspace($(AS.Shared))* %untyped_ptr
             }
             attributes #0 = { alwaysinline }
          """, "entry"), LLVMPtr{UInt32, AS.Shared}, Tuple{})
    CuDeviceArray{UInt32,1,AS.Shared}(ptr, (32,))
end

# initialization function, called automatically at the start of each kernel because
# there's no reliable way to detect uninitialized shared memory (see JuliaGPU/CUDA.jl#2008)
function initialize_rng_state()
    threadId = threadIdx().x + (threadIdx().y - 1i32) * blockDim().x +
                               (threadIdx().z - 1i32) * blockDim().x * blockDim().y
    warpId = (threadId - 1i32) >> 0x5 + 1i32  # fld1

    @inbounds global_random_keys()[warpId] = kernel_state().random_seed
    @inbounds global_random_counters()[warpId] = 0
end

# generators

using Random123: philox2x_round, philox2x_bumpkey

# GPU-compatible/optimized version of the generator from Random123.jl
struct Philox2x32{R} <: RandomNumbers.AbstractRNG{UInt64}
    # NOTE: the state is stored globally; see comments at the top of this file.
end

# default to 7 rounds; enough to pass BigCrush
@inline Philox2x32() = Philox2x32{7}()

@inline function Base.getproperty(rng::Philox2x32, field::Symbol)
    threadId = threadIdx().x + (threadIdx().y - 1i32) * blockDim().x +
                               (threadIdx().z - 1i32) * blockDim().x * blockDim().y
    warpId = (threadId - 1i32) >> 0x5 + 1i32  # fld1

    if field === :key
        @inbounds global_random_keys()[warpId]
    elseif field === :ctr1
        @inbounds global_random_counters()[warpId]
    elseif field === :ctr2
        blockId = blockIdx().x + (blockIdx().y - 1i32) * gridDim().x +
                                 (blockIdx().z - 1i32) * gridDim().x * gridDim().y
        globalId = threadId + (blockId - 1i32) * (blockDim().x * blockDim().y * blockDim().z)
        globalId%UInt32
    end::UInt32
end

@inline function Base.setproperty!(rng::Philox2x32, field::Symbol, x)
    threadId = threadIdx().x + (threadIdx().y - 1i32) * blockDim().x +
                               (threadIdx().z - 1i32) * blockDim().x * blockDim().y
    warpId = (threadId - 1i32) >> 0x5 + 1i32  # fld1

    if field === :key
        @inbounds global_random_keys()[warpId] = x
    elseif field === :ctr1
        @inbounds global_random_counters()[warpId] = x
    end
end

@device_override @inline Random.default_rng() = Philox2x32()

# default to Float32 on GPU (matches CUDA convention, avoids expensive FP64)
@device_override @inline Random.rand(rng::AbstractRNG) = Random.rand(rng, Float32)

"""
    Random.seed!(rng::Philox2x32, seed::Integer, [counter::Integer=0])

Seed the on-device Philox2x32 generator with an UInt32 number.
Should be called by at least one thread per warp.
"""
function Random.seed!(rng::Philox2x32, seed::Integer, counter::Integer=0)
    rng.key = seed % UInt32
    rng.ctr1 = counter
    return
end

if VERSION >= v"1.11-"
    # `Random.seed!(::AbstractRNG)` now passes a `nothing` seed value
    Random.seed!(rng::Philox2x32, seed::Nothing) =
        Random.seed!(rng, clock(UInt32))
else
    # ... where it used to call `Random_make_seed()`
    @device_override Random.make_seed() = clock(UInt32)
end

# seeding the implicit default RNG
if VERSION >= v"1.11-"
    @device_override Random.seed!(seed) =
        Random.seed!(Random.default_rng(), seed)
else
    @device_override Random.seed!(::Random._GLOBAL_RNG, seed) =
        Random.seed!(Random.default_rng(), seed)
end

# R rounds of Philox2x32, unrolled at compile time
@inline function philox2x_rounds(::Val{R}, ctr1::UInt32, ctr2::UInt32,
                                  key::UInt32) where R
    if R > 0                               ctr1, ctr2 = philox2x_round(ctr1, ctr2, key); end
    if R > 1  key = philox2x_bumpkey(key); ctr1, ctr2 = philox2x_round(ctr1, ctr2, key); end
    if R > 2  key = philox2x_bumpkey(key); ctr1, ctr2 = philox2x_round(ctr1, ctr2, key); end
    if R > 3  key = philox2x_bumpkey(key); ctr1, ctr2 = philox2x_round(ctr1, ctr2, key); end
    if R > 4  key = philox2x_bumpkey(key); ctr1, ctr2 = philox2x_round(ctr1, ctr2, key); end
    if R > 5  key = philox2x_bumpkey(key); ctr1, ctr2 = philox2x_round(ctr1, ctr2, key); end
    if R > 6  key = philox2x_bumpkey(key); ctr1, ctr2 = philox2x_round(ctr1, ctr2, key); end
    if R > 7  key = philox2x_bumpkey(key); ctr1, ctr2 = philox2x_round(ctr1, ctr2, key); end
    if R > 8  key = philox2x_bumpkey(key); ctr1, ctr2 = philox2x_round(ctr1, ctr2, key); end
    if R > 9  key = philox2x_bumpkey(key); ctr1, ctr2 = philox2x_round(ctr1, ctr2, key); end
    if R > 10 key = philox2x_bumpkey(key); ctr1, ctr2 = philox2x_round(ctr1, ctr2, key); end
    if R > 11 key = philox2x_bumpkey(key); ctr1, ctr2 = philox2x_round(ctr1, ctr2, key); end
    if R > 12 key = philox2x_bumpkey(key); ctr1, ctr2 = philox2x_round(ctr1, ctr2, key); end
    if R > 13 key = philox2x_bumpkey(key); ctr1, ctr2 = philox2x_round(ctr1, ctr2, key); end
    if R > 14 key = philox2x_bumpkey(key); ctr1, ctr2 = philox2x_round(ctr1, ctr2, key); end
    if R > 15 key = philox2x_bumpkey(key); ctr1, ctr2 = philox2x_round(ctr1, ctr2, key); end
    ctr1, ctr2
end

"""
    Random.rand(rng::Philox2x32, UInt64)

Generate 64 bits of random data using the on-device Philox2x32 generator.
"""
function Random.rand(rng::Philox2x32{R}, ::Type{UInt64}) where {R}
    ctr1, ctr2 = philox2x_rounds(Val(R), rng.ctr1, rng.ctr2, rng.key)

    # update the warp counter
    # NOTE: this performs the same update on every thread in the warp, but each warp writes
    #       to a unique location so the duplicate writes are innocuous
    # NOTE: this is not guaranteed to be visible in other kernels (JuliaGPU/CUDA.jl#2008)
    # XXX: what if this overflows? we can't increment ctr2. bump the key?
    rng.ctr1 += 1i32

    # NOTE: it's too expensive to keep both numbers around in case the user only wanted one,
    #       so just make our 2x32 generator return 64-bit numbers by default.
    return (ctr1 % UInt64) << 32 | (ctr2 % UInt64)
end



# normally distributed random numbers using Ziggurat algorithm
#
# copied from Base because we don't support its global tables

# a hacky method of exposing constant tables as constant GPU memory
function emit_constant_array(name::Symbol, data::AbstractArray{T}) where {T}
    @dispose ctx=Context() begin
        T_val = convert(LLVMType, T)
        T_ptr = convert(LLVMType, LLVMPtr{T,AS.Constant})

        # define function and get LLVM module
        llvm_f, _ = create_function(T_ptr)
        mod = LLVM.parent(llvm_f)

        # create a global memory global variable
        # TODO: global_var alignment?
        T_global = LLVM.ArrayType(T_val, length(data))
        # XXX: why can't we use a single name like emit_shmem
        gv = GlobalVariable(mod, T_global, "gpu_$(name)_data", AS.Constant)
        alignment!(gv, 16)
        linkage!(gv, LLVM.API.LLVMInternalLinkage)
        initializer!(gv, ConstantArray(data))

        # generate IR
        @dispose builder=IRBuilder() begin
            entry = BasicBlock(llvm_f, "entry")
            position!(builder, entry)

            ptr = gep!(builder, T_global, gv, [ConstantInt(0), ConstantInt(0)])

            untyped_ptr = bitcast!(builder, ptr, T_ptr)

            ret!(builder, untyped_ptr)
        end

        call_function(llvm_f, LLVMPtr{T,AS.Constant})
    end
end

for var in [:ke, :we, :fe]
    val = getfield(Random, var)
    gpu_var = Symbol("gpu_$var")
    arr_typ = :(CuDeviceArray{$(eltype(val)),$(ndims(val)),AS.Constant})
    @eval @inline @generated function $gpu_var()
        ptr = emit_constant_array($(QuoteNode(var)), $val)
        Expr(:call, $arr_typ, ptr, $(size(val)))
    end
end

## Box-Muller helpers
#
# Vendored from GPUArrays.jl, which uses them in its host-side Philox4x32-10
# batched randn kernel. Keep constants in sync when upstream tunes them.

using Base: FastMath

# unsigned int → uniform float in (0, 1), strictly positive

@inline u01(::Type{Float32}, u::UInt32) =
    fma(Float32(u), Float32(2)^(-32), Float32(2)^(-33))

# Bit-pattern construction avoids Float64(::UInt64) + FMA on consumer GPUs
# (FP64 throughput as low as 1:64). Low mantissa bit set so result ∈ (0, 1) —
# Box-Muller needs log(u) ≠ -Inf.
@inline u01(::Type{Float64}, u::UInt64) =
    reinterpret(Float64, ((u >> 12) | 0x1) | 0x3ff0000000000000) - 1.0

# Polynomial sincospi(Float32): branchless, stays in Float32 (Base.sincospi
# widens internally). Bottom 3 bits of u pick an octant (swap/negate); top
# 29 bits give the reduced argument (+0.5-biased so y ≠ 0).

const SP_F32 = (3.1415927f0, -5.167708f0, 2.5497673f0, -0.58907866f0)
const CP_F32 = (1.0f0, -4.934788f0, 4.057578f0, -1.3061346f0)

@inline function fast_sincospi(::Type{Float32}, u::UInt32)
    oct = (u % Int32) & Int32(7)
    y = fma(Float32(u & ~UInt32(7)), Float32(2)^(-34), Float32(2)^(-32))
    sp = y * evalpoly(y * y, SP_F32)
    cp = evalpoly(y * y, CP_F32)
    swap    = !iszero(oct & Int32(1))
    sin_neg = !iszero(oct & Int32(2))
    cos_neg = !iszero(oct & Int32(4))
    s_raw = ifelse(swap, cp, sp)
    c_raw = ifelse(swap, sp, cp)
    (ifelse(sin_neg, -s_raw, s_raw), ifelse(cos_neg, -c_raw, c_raw))
end

# Polynomial log(Float32), fdlibm-based. Consumes the raw UInt32 output; u01
# is folded into the first FMA so there's no intermediate float.

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

# Box-Muller: pair of uniforms → pair of standard normals

@inline function boxmuller(::Type{T}, u1::UInt32, u2::UInt32) where T <: Union{Float16,Float32}
    r = sqrt(-2f0 * fast_log(Float32, u2))
    s, c = fast_sincospi(Float32, u1)
    (T(r * s), T(r * c))
end

@inline function boxmuller(::Type{Float64}, u1::Float64, u2::Float64)
    r = sqrt(-2.0 * FastMath.log_fast(u1))
    s, c = sincospi(2 * u2)
    (r * s, r * c)
end


## randn — Box-Muller transform
#
# Uses Box-Muller instead of Ziggurat: rejection sampling would warp-diverge,
# and the Ziggurat tables aren't device-accessible.

# Specialization for Philox2x32: one Philox call produces exactly the pair of
# UInt32s Box-Muller needs, halving the Philox work vs the generic path.
@device_override @inline function Random.randn(rng::Philox2x32{R},
                                                ::Type{T}) where {R, T <: Union{Float16,Float32}}
    ctr1, ctr2 = philox2x_rounds(Val(R), rng.ctr1, rng.ctr2, rng.key)
    rng.ctr1 += 1i32
    n, _ = boxmuller(T, ctr1, ctr2)
    n
end

# Float64 fundamentally needs 64 bits of entropy per uniform, so 2 Philox
# calls. The u01 bit-trick avoids the expensive Float64(::UInt64) conversion.
@device_override @inline function Random.randn(rng::Philox2x32{R},
                                                ::Type{Float64}) where R
    u1 = u01(Float64, Random.rand(rng, UInt64))
    u2 = u01(Float64, Random.rand(rng, UInt64))
    n, _ = boxmuller(Float64, u1, u2)
    n
end

# Generic fallback for user-defined AbstractFloat types.
@device_override @inline function Random.randn(rng::AbstractRNG, ::Type{T}) where T <: AbstractFloat
    U1 = max(Random.rand(rng, T), floatmin(T))  # avoid log(0)
    U2 = Random.rand(rng, T)
    sqrt(T(-2) * FastMath.log_fast(U1)) * first(sincospi(T(2) * U2))
end

# untyped randn() defaults to Float32 on GPU
@device_override @inline Random.randn(rng::AbstractRNG) = Random.randn(rng, Float32)

## randexp

@device_override function Random.randexp(rng::AbstractRNG)
    while true
        ri = Random.rand(rng, Random.UInt52Raw()) % UInt64
        @inbounds begin
            ri &= 0x000fffffffffffff
            idx = ri & 0xFF
            x = ri*gpu_we()[idx+1]
            ri < gpu_ke()[idx+1] && return x # 98.9% of the time we return here 1st try
            result = randexp_unlikely(rng, idx, x)
            result !== nothing && return result
        end
    end
end

@noinline function randexp_unlikely(rng, idx, x)
    @inbounds if idx == 0
        return Random.ziggurat_exp_r - log(Random.rand(rng))
    elseif (gpu_fe()[idx] - gpu_fe()[idx+1])*Random.rand(rng) + gpu_fe()[idx+1] < exp(-x)
        return x # return from the triangular area
    else
        return # retry
    end
end
