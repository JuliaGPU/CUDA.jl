# math functionality

# we only use libdevice where needed. if possible, we go through LLVM instead,
# ideally relying on Julia's existing definitions.

@public fma, rsqrt, saturate, byte_perm, dp4a, assume
@public add_rn, add_rz, add_rm, add_rp
@public sub_rn, sub_rz, sub_rm, sub_rp
@public mul_rn, mul_rz, mul_rm, mul_rp
@public div_rn, div_rz, div_rm, div_rp
@public fma_rn, fma_rz, fma_rm, fma_rp

using Base: FastMath, @assume_effects

## helpers

within(lower, upper) = (val) -> lower <= val <= upper


## trigonometric

@device_override Base.cos(x::Float64) = ccall("extern __nv_cos", llvmcall, Cdouble, (Cdouble,), x)
@device_override Base.cos(x::Float32) = ccall("extern __nv_cosf", llvmcall, Cfloat, (Cfloat,), x)
@device_override FastMath.cos_fast(x::Float32) = ccall("extern __nv_fast_cosf", llvmcall, Cfloat, (Cfloat,), x)

@device_override Base.cospi(x::Float64) = ccall("extern __nv_cospi", llvmcall, Cdouble, (Cdouble,), x)
@device_override Base.cospi(x::Float32) = ccall("extern __nv_cospif", llvmcall, Cfloat, (Cfloat,), x)

@device_override Base.sin(x::Float64) = ccall("extern __nv_sin", llvmcall, Cdouble, (Cdouble,), x)
@device_override Base.sin(x::Float32) = ccall("extern __nv_sinf", llvmcall, Cfloat, (Cfloat,), x)
@device_override FastMath.sin_fast(x::Float32) = ccall("extern __nv_fast_sinf", llvmcall, Cfloat, (Cfloat,), x)

@device_override Base.sinpi(x::Float64) = ccall("extern __nv_sinpi", llvmcall, Cdouble, (Cdouble,), x)
@device_override Base.sinpi(x::Float32) = ccall("extern __nv_sinpif", llvmcall, Cfloat, (Cfloat,), x)

@device_override Base.tan(x::Float64) = ccall("extern __nv_tan", llvmcall, Cdouble, (Cdouble,), x)
@device_override Base.tan(x::Float32) = ccall("extern __nv_tanf", llvmcall, Cfloat, (Cfloat,), x)
@device_override FastMath.tan_fast(x::Float32) = ccall("extern __nv_fast_tanf", llvmcall, Cfloat, (Cfloat,), x)

@device_override function Base.sincos(x::Float64)
    s = Ref{Cdouble}()
    c = Ref{Cdouble}()
    ccall("extern __nv_sincos", llvmcall, Cvoid, (Cdouble, Ptr{Cdouble}, Ptr{Cdouble}), x, s, c)
    return (s[], c[])
end
@device_override function Base.sincos(x::Float32)
    s = Ref{Cfloat}()
    c = Ref{Cfloat}()
    ccall("extern __nv_sincosf", llvmcall, Cvoid, (Cfloat, Ptr{Cfloat}, Ptr{Cfloat}), x, s, c)
    return (s[], c[])
end
# Base has sincos_fast fall back to the native implementation which is presumed faster,
# but that is not the case compared to CUDA's intrinsics
@device_override FastMath.sincos_fast(x::Union{Float64,Float32}) = (FastMath.sin_fast(x), FastMath.cos_fast(x))

@device_override function Base.sincospi(x::Float64)
    s = Ref{Cdouble}()
    c = Ref{Cdouble}()
    ccall("extern __nv_sincospi", llvmcall, Cvoid, (Cdouble, Ptr{Cdouble}, Ptr{Cdouble}), x, s, c)
    return (s[], c[])
end
@device_override function Base.sincospi(x::Float32)
    s = Ref{Cfloat}()
    c = Ref{Cfloat}()
    ccall("extern __nv_sincospif", llvmcall, Cvoid, (Cfloat, Ptr{Cfloat}, Ptr{Cfloat}), x, s, c)
    return (s[], c[])
end


## inverse trigonometric

@device_override Base.acos(x::Float64) = ccall("extern __nv_acos", llvmcall, Cdouble, (Cdouble,), x)
@device_override Base.acos(x::Float32) = ccall("extern __nv_acosf", llvmcall, Cfloat, (Cfloat,), x)

@device_override Base.asin(x::Float64) = ccall("extern __nv_asin", llvmcall, Cdouble, (Cdouble,), x)
@device_override Base.asin(x::Float32) = ccall("extern __nv_asinf", llvmcall, Cfloat, (Cfloat,), x)

@device_override Base.atan(x::Float64) = ccall("extern __nv_atan", llvmcall, Cdouble, (Cdouble,), x)
@device_override Base.atan(x::Float32) = ccall("extern __nv_atanf", llvmcall, Cfloat, (Cfloat,), x)

@device_override Base.atan(x::Float64, y::Float64) = ccall("extern __nv_atan2", llvmcall, Cdouble, (Cdouble, Cdouble), x, y)
@device_override Base.atan(x::Float32, y::Float32) = ccall("extern __nv_atan2f", llvmcall, Cfloat, (Cfloat, Cfloat), x, y)

## hyperbolic

@device_override Base.cosh(x::Float64) = ccall("extern __nv_cosh", llvmcall, Cdouble, (Cdouble,), x)
@device_override Base.cosh(x::Float32) = ccall("extern __nv_coshf", llvmcall, Cfloat, (Cfloat,), x)

@device_override Base.sinh(x::Float64) = ccall("extern __nv_sinh", llvmcall, Cdouble, (Cdouble,), x)
@device_override Base.sinh(x::Float32) = ccall("extern __nv_sinhf", llvmcall, Cfloat, (Cfloat,), x)

@device_override Base.tanh(x::Float64) = ccall("extern __nv_tanh", llvmcall, Cdouble, (Cdouble,), x)
@device_override Base.tanh(x::Float32) = ccall("extern __nv_tanhf", llvmcall, Cfloat, (Cfloat,), x)
# Base.tanh(::Float16) is inherited from Julia (computes in Float32, which routes
# through __nv_tanhf). Hardware `tanh.approx.f16` is approximate, so it lives in
# FastMath — matching cuda_fp16.hpp where __htanh_approx is separate from __htanh.
@device_override function FastMath.tanh_fast(x::Float16)
    if compute_capability() >= sv"7.5"
        @asmcall("tanh.approx.f16 \$0, \$1;", "=h,h", Float16, Tuple{Float16}, x)
    else
        Float16(FastMath.tanh_fast(Float32(x)))
    end
end

## inverse hyperbolic

@device_override Base.acosh(x::Float64) = ccall("extern __nv_acosh", llvmcall, Cdouble, (Cdouble,), x)
@device_override Base.acosh(x::Float32) = ccall("extern __nv_acoshf", llvmcall, Cfloat, (Cfloat,), x)

@device_override Base.asinh(x::Float64) = ccall("extern __nv_asinh", llvmcall, Cdouble, (Cdouble,), x)
@device_override Base.asinh(x::Float32) = ccall("extern __nv_asinhf", llvmcall, Cfloat, (Cfloat,), x)

@device_override Base.atanh(x::Float64) = ccall("extern __nv_atanh", llvmcall, Cdouble, (Cdouble,), x)
@device_override Base.atanh(x::Float32) = ccall("extern __nv_atanhf", llvmcall, Cfloat, (Cfloat,), x)


## logarithmic

# Float16 strategy for log/exp variants: compute in Float32 using the approximate
# hardware path, then patch the handful of inputs (~2–5) where rounding the
# approximate result back to Float16 disagrees with the IEEE-correct value. The
# fixup table is the one from cuda_fp16.hpp (__hlog/__hexp/etc.); each entry pairs
# a Float16 input bit pattern with the ±1 ULP correction needed at that point.
# After the fixups, the Float16 output matches the IEEE-correct value bit-for-bit,
# so Julia semantics are preserved while still using the fast lg2.approx/ex2.approx
# hardware instead of libdevice's full-precision __nv_log*/__nv_exp*.

@device_override Base.log(x::Float64) = ccall("extern __nv_log", llvmcall, Cdouble, (Cdouble,), x)
@device_override Base.log(x::Float32) = ccall("extern __nv_logf", llvmcall, Cfloat, (Cfloat,), x)
@device_override function Base.log(h::Float16)
    f = @fastmath log(Float32(h))
    r = Float16(f)

    r = fma(Float16(h == reinterpret(Float16, 0x160D)), reinterpret(Float16, 0x9C00), r)
    r = fma(Float16(h == reinterpret(Float16, 0x3BFE)), reinterpret(Float16, 0x8010), r)
    r = fma(Float16(h == reinterpret(Float16, 0x3C0B)), reinterpret(Float16, 0x8080), r)
    r = fma(Float16(h == reinterpret(Float16, 0x6051)), reinterpret(Float16, 0x1C00), r)

    return r
end
@device_override FastMath.log_fast(x::Float32) = ccall("extern __nv_fast_logf", llvmcall, Cfloat, (Cfloat,), x)

@device_override Base.log10(x::Float64) = ccall("extern __nv_log10", llvmcall, Cdouble, (Cdouble,), x)
@device_override Base.log10(x::Float32) = ccall("extern __nv_log10f", llvmcall, Cfloat, (Cfloat,), x)
@device_override function Base.log10(h::Float16)
    f = @fastmath log10(Float32(h))
    r = Float16(f)

    r = fma(Float16(h == reinterpret(Float16, 0x338F)), reinterpret(Float16, 0x1000), r)
    r = fma(Float16(h == reinterpret(Float16, 0x33F8)), reinterpret(Float16, 0x9000), r)
    r = fma(Float16(h == reinterpret(Float16, 0x57E1)), reinterpret(Float16, 0x9800), r)
    r = fma(Float16(h == reinterpret(Float16, 0x719D)), reinterpret(Float16, 0x9C00), r)

    return r
end
@device_override FastMath.log10_fast(x::Float32) = ccall("extern __nv_fast_log10f", llvmcall, Cfloat, (Cfloat,), x)

@device_override Base.log1p(x::Float64) = ccall("extern __nv_log1p", llvmcall, Cdouble, (Cdouble,), x)
@device_override Base.log1p(x::Float32) = ccall("extern __nv_log1pf", llvmcall, Cfloat, (Cfloat,), x)

@device_override Base.log2(x::Float64) = ccall("extern __nv_log2", llvmcall, Cdouble, (Cdouble,), x)
@device_override Base.log2(x::Float32) = ccall("extern __nv_log2f", llvmcall, Cfloat, (Cfloat,), x)
@device_override function Base.log2(h::Float16)
    r = Float16(@fastmath log2(Float32(h)))

    # NB: log2 checks the *output* against the special-case pattern, not the input
    r = fma(Float16(r == reinterpret(Float16, 0xA2E2)), reinterpret(Float16, 0x8080), r)
    r = fma(Float16(r == reinterpret(Float16, 0xBF46)), reinterpret(Float16, 0x9400), r)

    return r
end
@device_override FastMath.log2_fast(x::Float32) = ccall("extern __nv_fast_log2f", llvmcall, Cfloat, (Cfloat,), x)

@device_function logb(x::Float64) = ccall("extern __nv_logb", llvmcall, Cdouble, (Cdouble,), x)
@device_function logb(x::Float32) = ccall("extern __nv_logbf", llvmcall, Cfloat, (Cfloat,), x)

@device_function ilogb(x::Float64) = ccall("extern __nv_ilogb", llvmcall, Int32, (Cdouble,), x)
@device_function ilogb(x::Float32) = ccall("extern __nv_ilogbf", llvmcall, Int32, (Cfloat,), x)


## exponential

@device_override Base.exp(x::Float64) = ccall("extern __nv_exp", llvmcall, Cdouble, (Cdouble,), x)
@device_override Base.exp(x::Float32) = ccall("extern __nv_expf", llvmcall, Cfloat, (Cfloat,), x)
@device_override function Base.exp(h::Float16)
    # exp(x) = exp2(x * log2(e)); the negative-zero addend keeps the fma from
    # collapsing into a plain multiply (matches cuda_fp16.hpp's hexp).
    f = fma(Float32(h), log2(Float32(ℯ)), -0f0)
    r = Float16(@fastmath exp2(f))

    r = fma(Float16(h == reinterpret(Float16, 0x1F79)), reinterpret(Float16, 0x9400), r)
    r = fma(Float16(h == reinterpret(Float16, 0x25CF)), reinterpret(Float16, 0x9400), r)
    r = fma(Float16(h == reinterpret(Float16, 0xC13B)), reinterpret(Float16, 0x0400), r)
    r = fma(Float16(h == reinterpret(Float16, 0xC1EF)), reinterpret(Float16, 0x0200), r)

    return r
end
@device_override FastMath.exp_fast(x::Float32) = ccall("extern __nv_fast_expf", llvmcall, Cfloat, (Cfloat,), x)

@device_override Base.exp2(x::Float64) = ccall("extern __nv_exp2", llvmcall, Cdouble, (Cdouble,), x)
@device_override Base.exp2(x::Float32) = ccall("extern __nv_exp2f", llvmcall, Cfloat, (Cfloat,), x)
@device_override function Base.exp2(h::Float16)
    f = @fastmath exp2(Float32(h))
    # one-ULP nudge: ex2.approx.ftz.f32 underestimates by ~½ ULP on average, so
    # bump up by 2^-24 of the result before rounding back to Float16
    return Float16(muladd(f, 2f0^-24, f))
end
@device_override FastMath.exp2_fast(x::Float64) = exp2(x)
@device_override FastMath.exp2_fast(x::Float32) =
    ccall("llvm.nvvm.ex2.approx.f", llvmcall, Float32, (Float32,), x)
@device_override function FastMath.exp2_fast(x::Float16)
    if compute_capability() >= sv"7.5"
        ccall("llvm.nvvm.ex2.approx.f16", llvmcall, Float16, (Float16,), x)
    else
        Float16(FastMath.exp2_fast(Float32(x)))
    end
end

@device_override Base.exp10(x::Float64) = ccall("extern __nv_exp10", llvmcall, Cdouble, (Cdouble,), x)
@device_override Base.exp10(x::Float32) = ccall("extern __nv_exp10f", llvmcall, Cfloat, (Cfloat,), x)
@device_override function Base.exp10(h::Float16)
    f = fma(Float32(h), log2(10f0), -0f0)
    r = Float16(@fastmath exp2(f))

    r = fma(Float16(h == reinterpret(Float16, 0x34DE)), reinterpret(Float16, 0x9800), r)
    r = fma(Float16(h == reinterpret(Float16, 0x9766)), reinterpret(Float16, 0x9000), r)
    r = fma(Float16(h == reinterpret(Float16, 0x9972)), reinterpret(Float16, 0x1000), r)
    r = fma(Float16(h == reinterpret(Float16, 0xA5C4)), reinterpret(Float16, 0x1000), r)
    r = fma(Float16(h == reinterpret(Float16, 0xBF0A)), reinterpret(Float16, 0x8100), r)

    return r
end
@device_override FastMath.exp10_fast(x::Float32) = ccall("extern __nv_fast_exp10f", llvmcall, Cfloat, (Cfloat,), x)

@device_override Base.expm1(x::Float64) = ccall("extern __nv_expm1", llvmcall, Cdouble, (Cdouble,), x)
@device_override Base.expm1(x::Float32) = ccall("extern __nv_expm1f", llvmcall, Cfloat, (Cfloat,), x)
@device_override Base.expm1(x::Float16) = Float16(CUDACore.expm1(Float32(x)))

@device_override Base.ldexp(x::Float64, y::Int32) = ccall("extern __nv_ldexp", llvmcall, Cdouble, (Cdouble, Int32), x, y)
@device_override Base.ldexp(x::Float32, y::Int32) = ccall("extern __nv_ldexpf", llvmcall, Cfloat, (Cfloat, Int32), x, y)


## integer handling (bit twiddling)

@device_function brev(x::Union{Int32, UInt32}) =   ccall("extern __nv_brev", llvmcall, UInt32, (UInt32,), x)
@device_function brev(x::Union{Int64, UInt64}) =   ccall("extern __nv_brevll", llvmcall, UInt64, (UInt64,), x)


@device_function clz(x::Union{Int32, UInt32}) =
    assume(within(UInt32(0), UInt32(32)),
           ccall("extern __nv_clz", llvmcall, Int32, (UInt32,), x))
@device_function clz(x::Union{Int64, UInt64}) =
    assume(within(UInt64(0), UInt64(64)),
           ccall("extern __nv_clzll", llvmcall, Int32, (UInt64,), x))

@device_function ffs(x::Union{Int32, UInt32}) =
    assume(within(UInt32(0), UInt32(32)),
           ccall("extern __nv_ffs", llvmcall, Int32, (UInt32,), x))
@device_function ffs(x::Union{Int64, UInt64}) =
    assume(within(UInt64(0), UInt64(64)),
           ccall("extern __nv_ffsll", llvmcall, Int32, (UInt64,), x))

@device_function function fns(mask::Union{Int32,UInt32}, base::Integer, offset::Integer=0)
    # Reinterpret the input mask instead of letting `ccall` convert them with a range check
    mask %= UInt32

    pos = ccall("llvm.nvvm.fns", llvmcall, Int32,
                (UInt32, Int32, Int32), mask, base, offset)
    assume(within(UInt32(0), UInt32(32)), pos)
end

@device_function popc(x::Union{Int32, UInt32}) =
    assume(within(UInt32(0), UInt32(32)),
           ccall("extern __nv_popc", llvmcall, Int32, (UInt32,), x))
@device_function popc(x::Union{Int64, UInt64}) =
    assume(within(UInt64(0), UInt64(64)),
           ccall("extern __nv_popcll", llvmcall, Int32, (UInt64,), x))

@device_function function byte_perm(x::Union{Int32, UInt32, Int16, UInt16, Int8, UInt8},
                                    y::Union{Int32, UInt32, Int16, UInt16, Int8, UInt8},
                                    z::Union{Int32, UInt32, Int16, UInt16, Int8, UInt8})
    # Reinterpret the input values instead of letting `ccall` convert them with a range check
    x %= UInt32
    y %= UInt32
    z %= UInt32
    ccall("extern __nv_byte_perm", llvmcall, Int32, (UInt32, UInt32, UInt32), x, y, z)
end

"""
    dp4a(a, b, c)

Packed 4-element int8 (or uint8) dot product with 32-bit accumulation, mapped to a single
PTX `dp4a` instruction on sm_61+.

The semantics depend on the signedness of `a` and `b`:

- `dp4a(a::Int32,  b::Int32,  c::Int32)  -> Int32`  — signed × signed
- `dp4a(a::Int32,  b::UInt32, c::Int32)  -> Int32`  — signed × unsigned
- `dp4a(a::UInt32, b::Int32,  c::Int32)  -> Int32`  — unsigned × signed
- `dp4a(a::UInt32, b::UInt32, c::UInt32) -> UInt32` — unsigned × unsigned

Each 32-bit argument `a` and `b` is interpreted as four packed 8-bit integers. The result
is `c + a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3]` where the individual byte
extractions respect the signed/unsigned interpretation of each operand.

!!! note
    Requires compute capability sm_61 or higher.
"""
function dp4a end

@static if LLVM.version() >= v"21"
    # LLVM 21 added @llvm.nvvm.idp4a.[us].[us]; prefer the intrinsic over inline PTX so
    # the instruction participates in optimization and instruction selection.
    @device_function function dp4a(a::Int32, b::Int32, c::Int32)
        require_sm_61()
        ccall("llvm.nvvm.idp4a.s.s", llvmcall, Int32, (Int32, Int32, Int32), a, b, c)
    end

    @device_function function dp4a(a::Int32, b::UInt32, c::Int32)
        require_sm_61()
        ccall("llvm.nvvm.idp4a.s.u", llvmcall, Int32, (Int32, UInt32, Int32), a, b, c)
    end

    @device_function function dp4a(a::UInt32, b::Int32, c::Int32)
        require_sm_61()
        ccall("llvm.nvvm.idp4a.u.s", llvmcall, Int32, (UInt32, Int32, Int32), a, b, c)
    end

    @device_function function dp4a(a::UInt32, b::UInt32, c::UInt32)
        require_sm_61()
        ccall("llvm.nvvm.idp4a.u.u", llvmcall, UInt32, (UInt32, UInt32, UInt32), a, b, c)
    end
else
    @device_function function dp4a(a::Int32, b::Int32, c::Int32)
        require_sm_61()
        @asmcall("dp4a.s32.s32 \$0, \$1, \$2, \$3;", "=r,r,r,r", false,
                 Int32, Tuple{Int32, Int32, Int32}, a, b, c)
    end

    @device_function function dp4a(a::Int32, b::UInt32, c::Int32)
        require_sm_61()
        @asmcall("dp4a.s32.u32 \$0, \$1, \$2, \$3;", "=r,r,r,r", false,
                 Int32, Tuple{Int32, UInt32, Int32}, a, b, c)
    end

    @device_function function dp4a(a::UInt32, b::Int32, c::Int32)
        require_sm_61()
        @asmcall("dp4a.u32.s32 \$0, \$1, \$2, \$3;", "=r,r,r,r", false,
                 Int32, Tuple{UInt32, Int32, Int32}, a, b, c)
    end

    @device_function function dp4a(a::UInt32, b::UInt32, c::UInt32)
        require_sm_61()
        @asmcall("dp4a.u32.u32 \$0, \$1, \$2, \$3;", "=r,r,r,r", false,
                 UInt32, Tuple{UInt32, UInt32, UInt32}, a, b, c)
    end
end


## floating-point handling

@device_function nearbyint(x::Float64) = ccall("extern __nv_nearbyint", llvmcall, Cdouble, (Cdouble,), x)
@device_function nearbyint(x::Float32) = ccall("extern __nv_nearbyintf", llvmcall, Cfloat, (Cfloat,), x)

@device_function nextafter(x::Float64, y::Float64) = ccall("extern __nv_nextafter", llvmcall, Cdouble, (Cdouble, Cdouble), x, y)
@device_function nextafter(x::Float32, y::Float32) = ccall("extern __nv_nextafterf", llvmcall, Cfloat, (Cfloat, Cfloat), x, y)


## roots and powers

# NVPTX has native `rsqrt.approx.{f32,f64}`; call the intrinsic directly. The
# obvious alternative, `@fastmath 1/sqrt(x)`, also lowers to `rsqrt.approx`
# (via `PTXRSqrtFastPass`), but is too aggressive wrt. fast-math behavior.
@device_function rsqrt(x::Float64) = ccall("llvm.nvvm.rsqrt.approx.d", llvmcall, Cdouble, (Cdouble,), x)
@device_function rsqrt(x::Float32) = ccall("llvm.nvvm.rsqrt.approx.f", llvmcall, Cfloat, (Cfloat,), x)
@device_function rsqrt(x::Float16) = Float16(rsqrt(Float32(x)))

@device_override Base.cbrt(x::Float64) = ccall("extern __nv_cbrt", llvmcall, Cdouble, (Cdouble,), x)
@device_override Base.cbrt(x::Float32) = ccall("extern __nv_cbrtf", llvmcall, Cfloat, (Cfloat,), x)

@device_function rcbrt(x::Float64) = ccall("extern __nv_rcbrt", llvmcall, Cdouble, (Cdouble,), x)
@device_function rcbrt(x::Float32) = ccall("extern __nv_rcbrtf", llvmcall, Cfloat, (Cfloat,), x)

@device_override Base.:(^)(x::Float64, y::Float64) = ccall("extern __nv_pow", llvmcall, Cdouble, (Cdouble, Cdouble), x, y)
@device_override Base.:(^)(x::Float32, y::Float32) = ccall("extern __nv_powf", llvmcall, Cfloat, (Cfloat, Cfloat), x, y)
@device_override FastMath.pow_fast(x::Float32, y::Float32) = ccall("extern __nv_fast_powf", llvmcall, Cfloat, (Cfloat, Cfloat), x, y)
# pow_fast: Base methods call llvm.powi which NVPTX cannot lower (#3065)
@device_override @assume_effects :foldable @inline function FastMath.pow_fast(x::Float64, y::Integer)
    y == -1 && return inv(x)
    y == 0 && return one(x)
    y == 1 && return x
    y == 2 && return x*x
    y == 3 && return x*x*x
    x ^ y  # no fast variant for Float64; uses __nv_powi
end
@device_override @assume_effects :foldable @inline function FastMath.pow_fast(x::Float32, y::Integer)
    y == -1 && return inv(x)
    y == 0 && return one(x)
    y == 1 && return x
    y == 2 && return x*x
    y == 3 && return x*x*x
    FastMath.pow_fast(x, Float32(y))  # uses __nv_fast_powf
end
@device_override @assume_effects :foldable @inline function FastMath.pow_fast(x::Float16, y::Integer)
    y == -1 && return inv(x)
    y == 0 && return one(x)
    y == 1 && return x
    y == 2 && return x*x
    y == 3 && return x*x*x
    Float16(FastMath.pow_fast(Float32(x), Float32(y)))
end
@device_override Base.:(^)(x::Float64, y::Int32) = ccall("extern __nv_powi", llvmcall, Cdouble, (Cdouble, Int32), x, y)
@device_override Base.:(^)(x::Float32, y::Int32) = ccall("extern __nv_powif", llvmcall, Cfloat, (Cfloat, Int32), x, y)
@device_override @assume_effects :foldable @inline function Base.:(^)(x::Float32, y::Int64)
    y == -1 && return inv(x)
    y == 0 && return one(x)
    y == 1 && return x
    y == 2 && return x*x
    y == 3 && return x*x*x
    x ^ Float32(y)
end
@device_override @assume_effects :foldable @inline function Base.:(^)(x::Float64, y::Int64)
    y == -1 && return inv(x)
    y == 0 && return one(x)
    y == 1 && return x
    y == 2 && return x*x
    y == 3 && return x*x*x
    x ^ Float64(y)
end

## rounding and selection

# TODO: differentiate in return type, map correctly
#@device_override Base.round(x::Float64) = ccall("extern __nv_llround", llvmcall, Int64, (Cdouble,), x)
#@device_override Base.round(x::Float32) = ccall("extern __nv_llroundf", llvmcall, Int64, (Cfloat,), x)
#@device_override Base.round(x::Float64) = ccall("extern __nv_round", llvmcall, Cdouble, (Cdouble,), x)
#@device_override Base.round(x::Float32) = ccall("extern __nv_roundf", llvmcall, Cfloat, (Cfloat,), x)

# TODO: differentiate in return type, map correctly
#@device_override Base.rint(x::Float64) = ccall("extern __nv_llrint", llvmcall, Int64, (Cdouble,), x)
#@device_override Base.rint(x::Float32) = ccall("extern __nv_llrintf", llvmcall, Int64, (Cfloat,), x)
#@device_override Base.rint(x::Float64) = ccall("extern __nv_rint", llvmcall, Cdouble, (Cdouble,), x)
#@device_override Base.rint(x::Float32) = ccall("extern __nv_rintf", llvmcall, Cfloat, (Cfloat,), x)

#@device_override Base.min(x::Int32, y::Int32) = ccall("extern __nv_min", llvmcall, Int32, (Int32, Int32), x, y)
#@device_override Base.min(x::Int64, y::Int64) = ccall("extern __nv_llmin", llvmcall, Int64, (Int64, Int64), x, y)
#@device_override Base.min(x::UInt32, y::UInt32) = convert(UInt32, ccall("extern __nv_umin", llvmcall, Int32, (Int32, Int32), x, y))
#@device_override Base.min(x::UInt64, y::UInt64) = convert(UInt64, ccall("extern __nv_ullmin", llvmcall, Int64, (Int64, Int64), x, y))
# JuliaGPU/CUDA.jl#2111: fmin semantics wrt. NaN and signed zeros don't match Julia's
#@device_override Base.min(x::Float64, y::Float64) = ccall("extern __nv_fmin", llvmcall, Cdouble, (Cdouble, Cdouble), x, y)
#@device_override Base.min(x::Float32, y::Float32) = ccall("extern __nv_fminf", llvmcall, Cfloat, (Cfloat, Cfloat), x, y)
# Julia's floating-point min/max match IEEE 754-2019 minimum/maximum, i.e.,
# `llvm.minimum`/`llvm.maximum`, which the external back-end legalizes for every
# subtarget: native min.NaN/max.NaN instructions on sm_80+, an expansion with
# NaN/signed-zero fix-ups elsewhere. Don't be tempted to use `llvm.minnum`
# (libdevice's `__nv_fmin`) with a NaN fix-up instead: its loose signed-zero
# semantics leak into constant folding, e.g., folding `min(0.0, -0.0)` to
# `0.0` where the host returns `-0.0`.
# Julia 1.12+ lowers `Base.min` to `llvm.minimum` by itself; keep the overrides
# for uniform codegen on older versions.
@device_override Base.min(x::Float32, y::Float32) =
    ccall("llvm.minimum.f32", llvmcall, Float32, (Float32, Float32), x, y)
@device_override Base.min(x::Float64, y::Float64) =
    ccall("llvm.minimum.f64", llvmcall, Float64, (Float64, Float64), x, y)

#@device_override Base.max(x::Int32, y::Int32) = ccall("extern __nv_max", llvmcall, Int32, (Int32, Int32), x, y)
#@device_override Base.max(x::Int64, y::Int64) = ccall("extern __nv_llmax", llvmcall, Int64, (Int64, Int64), x, y)
#@device_override Base.max(x::UInt32, y::UInt32) = convert(UInt32, ccall("extern __nv_umax", llvmcall, Int32, (Int32, Int32), x, y))
#@device_override Base.max(x::UInt64, y::UInt64) = convert(UInt64, ccall("extern __nv_ullmax", llvmcall, Int64, (Int64, Int64), x, y))
# JuliaGPU/CUDA.jl#2111: fmax semantics wrt. NaN and signed zeros don't match Julia's
#@device_override Base.max(x::Float64, y::Float64) = ccall("extern __nv_fmax", llvmcall, Cdouble, (Cdouble, Cdouble), x, y)
#@device_override Base.max(x::Float32, y::Float32) = ccall("extern __nv_fmaxf", llvmcall, Cfloat, (Cfloat, Cfloat), x, y)
@device_override Base.max(x::Float32, y::Float32) =
    ccall("llvm.maximum.f32", llvmcall, Float32, (Float32, Float32), x, y)
@device_override Base.max(x::Float64, y::Float64) =
    ccall("llvm.maximum.f64", llvmcall, Float64, (Float64, Float64), x, y)

# Base's AbstractFloat minmax simply calls min/max, but Julia 1.10/1.11 had
# open-coded definitions for Float32/Float64; override for uniform codegen.
@device_override Base.minmax(x::Float32, y::Float32) = min(x, y), max(x, y)
@device_override Base.minmax(x::Float64, y::Float64) = min(x, y), max(x, y)

@device_function saturate(x::Float32) = ccall("extern __nv_saturatef", llvmcall, Cfloat, (Cfloat,), x)


## division and remainder

@device_override Base.rem(x::Float64, y::Float64) = ccall("extern __nv_fmod", llvmcall, Cdouble, (Cdouble, Cdouble), x, y)
@device_override Base.rem(x::Float32, y::Float32) = ccall("extern __nv_fmodf", llvmcall, Cfloat, (Cfloat, Cfloat), x, y)
@device_override Base.rem(x::Float16, y::Float16) = Float16(rem(Float32(x), Float32(y)))

@device_override Base.rem(x::Float64, y::Float64, ::RoundingMode{:Nearest}) = ccall("extern __nv_remainder", llvmcall, Cdouble, (Cdouble, Cdouble), x, y)
@device_override Base.rem(x::Float32, y::Float32, ::RoundingMode{:Nearest}) = ccall("extern __nv_remainderf", llvmcall, Cfloat, (Cfloat, Cfloat), x, y)
@device_override Base.rem(x::Float16, y::Float16, ::RoundingMode{:Nearest}) = Float16(rem(Float32(x), Float32(y), RoundNearest))

# `Base.FastMath.inv_fast(::AbstractFloat)` is unimplemented upstream (only
# `Complex` has a method) and the catch-all fallback drops `afn`
@device_override FastMath.inv_fast(x::Union{Float16, Float32, Float64}) =
    FastMath.div_fast(one(x), x)


## distributions

# TODO: override StatsFun.jl?

@device_function normcdf(x::Float64) = ccall("extern __nv_normcdf", llvmcall, Cdouble, (Cdouble,), x)
@device_function normcdf(x::Float32) = ccall("extern __nv_normcdff", llvmcall, Cfloat, (Cfloat,), x)

@device_function normcdfinv(x::Float64) = ccall("extern __nv_normcdfinv", llvmcall, Cdouble, (Cdouble,), x)
@device_function normcdfinv(x::Float32) = ccall("extern __nv_normcdfinvf", llvmcall, Cfloat, (Cfloat,), x)



#
# Unsorted
#

@device_override Base.hypot(x::Float64, y::Float64) = ccall("extern __nv_hypot", llvmcall, Cdouble, (Cdouble, Cdouble), x, y)
@device_override Base.hypot(x::Float32, y::Float32) = ccall("extern __nv_hypotf", llvmcall, Cfloat, (Cfloat, Cfloat), x, y)

# `Base.fma(::Float16,...)` branches on `jl_have_fma`
@device_override Base.fma(x::Float16, y::Float16, z::Float16) =
    ccall("llvm.fma.f16", llvmcall, Float16, (Float16, Float16, Float16), x, y, z)

# `Base.muladd(x, y, z) = fma(x, y, z)` is the natural choice on GPU: NVPTX
# always lowers `llvm.fmuladd.fXX` to `fma.rn`, and routing through
# `llvm.fmuladd` (rather than Julia's default `fmul contract + fadd contract`)
# keeps the fusion robust under vectorization (per JuliaGPU/CUDA.jl#3149).
@device_override Base.muladd(x::Float64, y::Float64, z::Float64) =
    ccall("llvm.fmuladd.f64", llvmcall, Cdouble, (Cdouble, Cdouble, Cdouble), x, y, z)
@device_override Base.muladd(x::Float32, y::Float32, z::Float32) =
    ccall("llvm.fmuladd.f32", llvmcall, Cfloat, (Cfloat, Cfloat, Cfloat), x, y, z)
@device_override Base.muladd(x::Float16, y::Float16, z::Float16) =
    ccall("llvm.fmuladd.f16", llvmcall, Float16, (Float16, Float16, Float16), x, y, z)

# Directed rounding for binary arithmetic and fma. NVPTX exposes
# `{add,mul,div,fma}.{rn,rz,rm,rp}.{f32,f64}` directly; there is no `sub`
# intrinsic, so subtraction reuses add(x, -y) (negation is bit-exact for IEEE
# floats). Names match the CUDA C intrinsics (`__fadd_rn`, `__dadd_rn`, ...) so
# users porting from CUDA C can find the corresponding Julia method by dropping
# the `__f`/`__d` prefix. We don't extend `Base.fma` / `Base.:+` etc. with a
# `RoundingMode` argument because Base has no precedent for per-call rounding
# on hardware floats (BigFloat reads a thread-local mode instead).
for rnd in ("rn", "rz", "rm", "rp")
    for op in (:add, :mul, :div)
        fname = Symbol(op, :_, rnd)
        intrinsic_f = "llvm.nvvm.$(op).$(rnd).f"
        intrinsic_d = "llvm.nvvm.$(op).$(rnd).d"
        @eval @device_function $fname(x::Float32, y::Float32) =
            ccall($intrinsic_f, llvmcall, Cfloat, (Cfloat, Cfloat), x, y)
        @eval @device_function $fname(x::Float64, y::Float64) =
            ccall($intrinsic_d, llvmcall, Cdouble, (Cdouble, Cdouble), x, y)
    end

    # NVPTX has no sub.<rnd> intrinsic; reuse add with negated y.
    sub_fname = Symbol(:sub_, rnd)
    add_fname = Symbol(:add_, rnd)
    @eval @device_function $sub_fname(x::T, y::T) where {T<:Union{Float32, Float64}} =
        $add_fname(x, -y)

    fma_fname = Symbol(:fma_, rnd)
    intrinsic_f = "llvm.nvvm.fma.$(rnd).f"
    intrinsic_d = "llvm.nvvm.fma.$(rnd).d"
    @eval @device_function $fma_fname(x::Float32, y::Float32, z::Float32) =
        ccall($intrinsic_f, llvmcall, Cfloat, (Cfloat, Cfloat, Cfloat), x, y, z)
    @eval @device_function $fma_fname(x::Float64, y::Float64, z::Float64) =
        ccall($intrinsic_d, llvmcall, Cdouble, (Cdouble, Cdouble, Cdouble), x, y, z)
end

@device_function sad(x::Int32, y::Int32, z::Int32) = ccall("extern __nv_sad", llvmcall, Int32, (Int32, Int32, Int32), x, y, z)
@device_function sad(x::UInt32, y::UInt32, z::UInt32) = convert(UInt32, ccall("extern __nv_usad", llvmcall, Int32, (Int32, Int32, Int32), x, y, z))

@device_function dim(x::Float64, y::Float64) = ccall("extern __nv_fdim", llvmcall, Cdouble, (Cdouble, Cdouble), x, y)
@device_function dim(x::Float32, y::Float32) = ccall("extern __nv_fdimf", llvmcall, Cfloat, (Cfloat, Cfloat), x, y)

@device_function mul24(x::Int32, y::Int32) = ccall("extern __nv_mul24", llvmcall, Int32, (Int32, Int32), x, y)
@device_function mul24(x::UInt32, y::UInt32) = convert(UInt32, ccall("extern __nv_umul24", llvmcall, Int32, (Int32, Int32), x, y))

@device_function mul64hi(x::Int64, y::Int64) = ccall("extern __nv_mul64hi", llvmcall, Int64, (Int64, Int64), x, y)
@device_function mul64hi(x::UInt64, y::UInt64) = convert(UInt64, ccall("extern __nv_umul64hi", llvmcall, Int64, (Int64, Int64), x, y))
@device_function mulhi(x::Int32, y::Int32) = ccall("extern __nv_mulhi", llvmcall, Int32, (Int32, Int32), x, y)
@device_function mulhi(x::UInt32, y::UInt32) = convert(UInt32, ccall("extern __nv_umulhi", llvmcall, Int32, (Int32, Int32), x, y))

@device_function hadd(x::Int32, y::Int32) = ccall("extern __nv_hadd", llvmcall, Int32, (Int32, Int32), x, y)
@device_function hadd(x::UInt32, y::UInt32) = convert(UInt32, ccall("extern __nv_uhadd", llvmcall, Int32, (Int32, Int32), x, y))

@device_function rhadd(x::Int32, y::Int32) = ccall("extern __nv_rhadd", llvmcall, Int32, (Int32, Int32), x, y)
@device_function rhadd(x::UInt32, y::UInt32) = convert(UInt32, ccall("extern __nv_urhadd", llvmcall, Int32, (Int32, Int32), x, y))

@device_function scalbn(x::Float64, y::Int32) = ccall("extern __nv_scalbn", llvmcall, Cdouble, (Cdouble, Int32), x, y)
@device_function scalbn(x::Float32, y::Int32) = ccall("extern __nv_scalbnf", llvmcall, Cfloat, (Cfloat, Int32), x, y)

@device_function norm3df(x::Float32, y::Float32, z::Float32) = ccall("extern __nv_norm3df", llvmcall, Cfloat, (Cfloat, Cfloat, Cfloat), x, y, z)
@device_function rnorm3df(x::Float32, y::Float32, z::Float32) = ccall("extern __nv_rnorm3df", llvmcall, Cfloat, (Cfloat, Cfloat, Cfloat), x, y, z)
