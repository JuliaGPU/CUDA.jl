# math functionality

using Base: FastMath

using SpecialFunctions


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

# TODO: enable once PTX > 7.0 is supported
# @device_override Base.tanh(x::Float16) = @asmcall("tanh.approx.f16 \$0, \$1", "=h,h", Float16, Tuple{Float16}, x)


## inverse hyperbolic

@device_override Base.acosh(x::Float64) = ccall("extern __nv_acosh", llvmcall, Cdouble, (Cdouble,), x)
@device_override Base.acosh(x::Float32) = ccall("extern __nv_acoshf", llvmcall, Cfloat, (Cfloat,), x)

@device_override Base.asinh(x::Float64) = ccall("extern __nv_asinh", llvmcall, Cdouble, (Cdouble,), x)
@device_override Base.asinh(x::Float32) = ccall("extern __nv_asinhf", llvmcall, Cfloat, (Cfloat,), x)

@device_override Base.atanh(x::Float64) = ccall("extern __nv_atanh", llvmcall, Cdouble, (Cdouble,), x)
@device_override Base.atanh(x::Float32) = ccall("extern __nv_atanhf", llvmcall, Cfloat, (Cfloat,), x)


## logarithmic

@device_override Base.log(x::Float64) = ccall("extern __nv_log", llvmcall, Cdouble, (Cdouble,), x)
@device_override Base.log(x::Float32) = ccall("extern __nv_logf", llvmcall, Cfloat, (Cfloat,), x)
@device_override FastMath.log_fast(x::Float32) = ccall("extern __nv_fast_logf", llvmcall, Cfloat, (Cfloat,), x)

@device_override Base.log10(x::Float64) = ccall("extern __nv_log10", llvmcall, Cdouble, (Cdouble,), x)
@device_override Base.log10(x::Float32) = ccall("extern __nv_log10f", llvmcall, Cfloat, (Cfloat,), x)
@device_override FastMath.log10_fast(x::Float32) = ccall("extern __nv_fast_log10f", llvmcall, Cfloat, (Cfloat,), x)

@device_override Base.log1p(x::Float64) = ccall("extern __nv_log1p", llvmcall, Cdouble, (Cdouble,), x)
@device_override Base.log1p(x::Float32) = ccall("extern __nv_log1pf", llvmcall, Cfloat, (Cfloat,), x)

@device_override Base.log2(x::Float64) = ccall("extern __nv_log2", llvmcall, Cdouble, (Cdouble,), x)
@device_override Base.log2(x::Float32) = ccall("extern __nv_log2f", llvmcall, Cfloat, (Cfloat,), x)
@device_override FastMath.log2_fast(x::Float32) = ccall("extern __nv_fast_log2f", llvmcall, Cfloat, (Cfloat,), x)

@device_function logb(x::Float64) = ccall("extern __nv_logb", llvmcall, Cdouble, (Cdouble,), x)
@device_function logb(x::Float32) = ccall("extern __nv_logbf", llvmcall, Cfloat, (Cfloat,), x)

@device_function ilogb(x::Float64) = ccall("extern __nv_ilogb", llvmcall, Int32, (Cdouble,), x)
@device_function ilogb(x::Float32) = ccall("extern __nv_ilogbf", llvmcall, Int32, (Cfloat,), x)


## exponential

@device_override Base.exp(x::Float64) = ccall("extern __nv_exp", llvmcall, Cdouble, (Cdouble,), x)
@device_override Base.exp(x::Float32) = ccall("extern __nv_expf", llvmcall, Cfloat, (Cfloat,), x)
@device_override FastMath.exp_fast(x::Float32) = ccall("extern __nv_fast_expf", llvmcall, Cfloat, (Cfloat,), x)

@device_override Base.exp2(x::Float64) = ccall("extern __nv_exp2", llvmcall, Cdouble, (Cdouble,), x)
@device_override Base.exp2(x::Float32) = ccall("extern __nv_exp2f", llvmcall, Cfloat, (Cfloat,), x)
# TODO: enable once PTX > 7.0 is supported
# @device_override Base.exp2(x::Float16) = @asmcall("ex2.approx.f16 \$0, \$1", "=h,h", Float16, Tuple{Float16}, x)

@device_override Base.exp10(x::Float64) = ccall("extern __nv_exp10", llvmcall, Cdouble, (Cdouble,), x)
@device_override Base.exp10(x::Float32) = ccall("extern __nv_exp10f", llvmcall, Cfloat, (Cfloat,), x)
@device_override FastMath.exp10_fast(x::Float32) = ccall("extern __nv_fast_exp10f", llvmcall, Cfloat, (Cfloat,), x)

@device_override Base.expm1(x::Float64) = ccall("extern __nv_expm1", llvmcall, Cdouble, (Cdouble,), x)
@device_override Base.expm1(x::Float32) = ccall("extern __nv_expm1f", llvmcall, Cfloat, (Cfloat,), x)
@device_override Base.expm1(x::Float16) = Float16(CUDA.expm1(Float32(x)))

@device_override Base.ldexp(x::Float64, y::Int32) = ccall("extern __nv_ldexp", llvmcall, Cdouble, (Cdouble, Int32), x, y)
@device_override Base.ldexp(x::Float32, y::Int32) = ccall("extern __nv_ldexpf", llvmcall, Cfloat, (Cfloat, Int32), x, y)

## error

@device_override SpecialFunctions.erf(x::Float64) = ccall("extern __nv_erf", llvmcall, Cdouble, (Cdouble,), x)
@device_override SpecialFunctions.erf(x::Float32) = ccall("extern __nv_erff", llvmcall, Cfloat, (Cfloat,), x)

@device_override SpecialFunctions.erfinv(x::Float64) = ccall("extern __nv_erfinv", llvmcall, Cdouble, (Cdouble,), x)
@device_override SpecialFunctions.erfinv(x::Float32) = ccall("extern __nv_erfinvf", llvmcall, Cfloat, (Cfloat,), x)

@device_override SpecialFunctions.erfc(x::Float64) = ccall("extern __nv_erfc", llvmcall, Cdouble, (Cdouble,), x)
@device_override SpecialFunctions.erfc(x::Float32) = ccall("extern __nv_erfcf", llvmcall, Cfloat, (Cfloat,), x)

@device_override SpecialFunctions.erfcinv(x::Float64) = ccall("extern __nv_erfcinv", llvmcall, Cdouble, (Cdouble,), x)
@device_override SpecialFunctions.erfcinv(x::Float32) = ccall("extern __nv_erfcinvf", llvmcall, Cfloat, (Cfloat,), x)

@device_override SpecialFunctions.erfcx(x::Float64) = ccall("extern __nv_erfcx", llvmcall, Cdouble, (Cdouble,), x)
@device_override SpecialFunctions.erfcx(x::Float32) = ccall("extern __nv_erfcxf", llvmcall, Cfloat, (Cfloat,), x)


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

@device_function popc(x::Union{Int32, UInt32}) =
    assume(within(UInt32(0), UInt32(32)),
           ccall("extern __nv_popc", llvmcall, Int32, (UInt32,), x))
@device_function popc(x::Union{Int64, UInt64}) =
    assume(within(UInt64(0), UInt64(64)),
           ccall("extern __nv_popcll", llvmcall, Int32, (UInt64,), x))

@device_function byte_perm(x::Union{Int32, UInt32}, y::Union{Int32, UInt32}, z::Union{Int32, UInt32}) =
    ccall("extern __nv_byte_perm", llvmcall, Int32, (UInt32, UInt32, UInt32), x, y, z)


## floating-point handling

@device_override Base.isfinite(x::Float32) = (ccall("extern __nv_finitef", llvmcall, Int32, (Cfloat,), x)) != 0
@device_override Base.isfinite(x::Float64) = (ccall("extern __nv_isfinited", llvmcall, Int32, (Cdouble,), x)) != 0

@device_override Base.isinf(x::Float64) = (ccall("extern __nv_isinfd", llvmcall, Int32, (Cdouble,), x)) != 0
@device_override Base.isinf(x::Float32) = (ccall("extern __nv_isinff", llvmcall, Int32, (Cfloat,), x)) != 0

@device_override Base.isnan(x::Float64) = (ccall("extern __nv_isnand", llvmcall, Int32, (Cdouble,), x)) != 0
@device_override Base.isnan(x::Float32) = (ccall("extern __nv_isnanf", llvmcall, Int32, (Cfloat,), x)) != 0

@device_function nearbyint(x::Float64) = ccall("extern __nv_nearbyint", llvmcall, Cdouble, (Cdouble,), x)
@device_function nearbyint(x::Float32) = ccall("extern __nv_nearbyintf", llvmcall, Cfloat, (Cfloat,), x)

@device_function nextafter(x::Float64, y::Float64) = ccall("extern __nv_nextafter", llvmcall, Cdouble, (Cdouble, Cdouble), x, y)
@device_function nextafter(x::Float32, y::Float32) = ccall("extern __nv_nextafterf", llvmcall, Cfloat, (Cfloat, Cfloat), x, y)


## sign handling

@device_override Base.signbit(x::Float64) = (ccall("extern __nv_signbitd", llvmcall, Int32, (Cdouble,), x)) != 0
@device_override Base.signbit(x::Float32) = (ccall("extern __nv_signbitf", llvmcall, Int32, (Cfloat,), x)) != 0

@device_override Base.copysign(x::Float64, y::Float64) = ccall("extern __nv_copysign", llvmcall, Cdouble, (Cdouble, Cdouble), x, y)
@device_override Base.copysign(x::Float32, y::Float32) = ccall("extern __nv_copysignf", llvmcall, Cfloat, (Cfloat, Cfloat), x, y)

@device_override Base.abs(x::Int32) =   ccall("extern __nv_abs", llvmcall, Int32, (Int32,), x)
@device_override Base.abs(f::Float64) = ccall("extern __nv_fabs", llvmcall, Cdouble, (Cdouble,), f)
@device_override Base.abs(f::Float32) = ccall("extern __nv_fabsf", llvmcall, Cfloat, (Cfloat,), f)
# TODO: enable once PTX > 7.0 is supported
# @device_override Base.abs(x::Float16) = @asmcall("abs.f16 \$0, \$1", "=h,h", Float16, Tuple{Float16}, x)
@device_override Base.abs(x::Int64) =   ccall("extern __nv_llabs", llvmcall, Int64, (Int64,), x)

## roots and powers

@device_override Base.sqrt(x::Float64) = ccall("extern __nv_sqrt", llvmcall, Cdouble, (Cdouble,), x)
@device_override Base.sqrt(x::Float32) = ccall("extern __nv_sqrtf", llvmcall, Cfloat, (Cfloat,), x)

@device_function rsqrt(x::Float64) = ccall("extern __nv_rsqrt", llvmcall, Cdouble, (Cdouble,), x)
@device_function rsqrt(x::Float32) = ccall("extern __nv_rsqrtf", llvmcall, Cfloat, (Cfloat,), x)
@device_function rsqrt(x::Float16) = Float16(rsqrt(Float32(x)))

@device_override Base.cbrt(x::Float64) = ccall("extern __nv_cbrt", llvmcall, Cdouble, (Cdouble,), x)
@device_override Base.cbrt(x::Float32) = ccall("extern __nv_cbrtf", llvmcall, Cfloat, (Cfloat,), x)

@device_function rcbrt(x::Float64) = ccall("extern __nv_rcbrt", llvmcall, Cdouble, (Cdouble,), x)
@device_function rcbrt(x::Float32) = ccall("extern __nv_rcbrtf", llvmcall, Cfloat, (Cfloat,), x)

@device_override Base.:(^)(x::Float64, y::Float64) = ccall("extern __nv_pow", llvmcall, Cdouble, (Cdouble, Cdouble), x, y)
@device_override Base.:(^)(x::Float32, y::Float32) = ccall("extern __nv_powf", llvmcall, Cfloat, (Cfloat, Cfloat), x, y)
@device_override FastMath.pow_fast(x::Float32, y::Float32) = ccall("extern __nv_fast_powf", llvmcall, Cfloat, (Cfloat, Cfloat), x, y)
@device_override Base.:(^)(x::Float64, y::Int32) = ccall("extern __nv_powi", llvmcall, Cdouble, (Cdouble, Int32), x, y)
@device_override Base.:(^)(x::Float32, y::Int32) = ccall("extern __nv_powif", llvmcall, Cfloat, (Cfloat, Int32), x, y)
@device_override @inline function Base.:(^)(x::Float32, y::Int64)
    y == -1 && return inv(x)
    y == 0 && return one(x)
    y == 1 && return x
    y == 2 && return x*x
    y == 3 && return x*x*x
    x ^ Float32(y)
end
@device_override @inline function Base.:(^)(x::Float64, y::Int64)
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

@device_override Base.trunc(x::Float64) = ccall("extern __nv_trunc", llvmcall, Cdouble, (Cdouble,), x)
@device_override Base.trunc(x::Float32) = ccall("extern __nv_truncf", llvmcall, Cfloat, (Cfloat,), x)

@device_override Base.ceil(x::Float64) = ccall("extern __nv_ceil", llvmcall, Cdouble, (Cdouble,), x)
@device_override Base.ceil(x::Float32) = ccall("extern __nv_ceilf", llvmcall, Cfloat, (Cfloat,), x)

@device_override Base.floor(f::Float64) = ccall("extern __nv_floor", llvmcall, Cdouble, (Cdouble,), f)
@device_override Base.floor(f::Float32) = ccall("extern __nv_floorf", llvmcall, Cfloat, (Cfloat,), f)

#@device_override Base.min(x::Int32, y::Int32) = ccall("extern __nv_min", llvmcall, Int32, (Int32, Int32), x, y)
#@device_override Base.min(x::Int64, y::Int64) = ccall("extern __nv_llmin", llvmcall, Int64, (Int64, Int64), x, y)
#@device_override Base.min(x::UInt32, y::UInt32) = convert(UInt32, ccall("extern __nv_umin", llvmcall, Int32, (Int32, Int32), x, y))
#@device_override Base.min(x::UInt64, y::UInt64) = convert(UInt64, ccall("extern __nv_ullmin", llvmcall, Int64, (Int64, Int64), x, y))
@device_override Base.min(x::Float64, y::Float64) = ccall("extern __nv_fmin", llvmcall, Cdouble, (Cdouble, Cdouble), x, y)
@device_override Base.min(x::Float32, y::Float32) = ccall("extern __nv_fminf", llvmcall, Cfloat, (Cfloat, Cfloat), x, y)

#@device_override Base.max(x::Int32, y::Int32) = ccall("extern __nv_max", llvmcall, Int32, (Int32, Int32), x, y)
#@device_override Base.max(x::Int64, y::Int64) = ccall("extern __nv_llmax", llvmcall, Int64, (Int64, Int64), x, y)
#@device_override Base.max(x::UInt32, y::UInt32) = convert(UInt32, ccall("extern __nv_umax", llvmcall, Int32, (Int32, Int32), x, y))
#@device_override Base.max(x::UInt64, y::UInt64) = convert(UInt64, ccall("extern __nv_ullmax", llvmcall, Int64, (Int64, Int64), x, y))
@device_override Base.max(x::Float64, y::Float64) = ccall("extern __nv_fmax", llvmcall, Cdouble, (Cdouble, Cdouble), x, y)
@device_override Base.max(x::Float32, y::Float32) = ccall("extern __nv_fmaxf", llvmcall, Cfloat, (Cfloat, Cfloat), x, y)

@device_function saturate(x::Float32) = ccall("extern __nv_saturatef", llvmcall, Cfloat, (Cfloat,), x)


## division and remainder

# NOTE: CUDA follows fmod, which behaves differently than Base.mod for negative numbers
#@device_override Base.mod(x::Float64, y::Float64) = ccall("extern __nv_fmod", llvmcall, Cdouble, (Cdouble, Cdouble), x, y)
#@device_override Base.mod(x::Float32, y::Float32) = ccall("extern __nv_fmodf", llvmcall, Cfloat, (Cfloat, Cfloat), x, y)

@device_override Base.rem(x::Float64, y::Float64) = ccall("extern __nv_remainder", llvmcall, Cdouble, (Cdouble, Cdouble), x, y)
@device_override Base.rem(x::Float32, y::Float32) = ccall("extern __nv_remainderf", llvmcall, Cfloat, (Cfloat, Cfloat), x, y)
@device_override Base.rem(x::Float16, y::Float16) = Float16(rem(Float32(x), Float32(y)))

@device_override FastMath.div_fast(x::Float32, y::Float32) = ccall("extern __nv_fast_fdividef", llvmcall, Cfloat, (Cfloat, Cfloat), x, y)


## gamma function

@device_override SpecialFunctions.lgamma(x::Float64) = ccall("extern __nv_lgamma", llvmcall, Cdouble, (Cdouble,), x)
@device_override SpecialFunctions.lgamma(x::Float32) = ccall("extern __nv_lgammaf", llvmcall, Cfloat, (Cfloat,), x)

@device_function tgamma(x::Float64) = ccall("extern __nv_tgamma", llvmcall, Cdouble, (Cdouble,), x)
@device_function tgamma(x::Float32) = ccall("extern __nv_tgammaf", llvmcall, Cfloat, (Cfloat,), x)


## Bessel

@device_override SpecialFunctions.besselj0(x::Float64) = ccall("extern __nv_j0", llvmcall, Cdouble, (Cdouble,), x)
@device_override SpecialFunctions.besselj0(x::Float32) = ccall("extern __nv_j0f", llvmcall, Cfloat, (Cfloat,), x)

@device_override SpecialFunctions.besselj1(x::Float64) = ccall("extern __nv_j1", llvmcall, Cdouble, (Cdouble,), x)
@device_override SpecialFunctions.besselj1(x::Float32) = ccall("extern __nv_j1f", llvmcall, Cfloat, (Cfloat,), x)

@device_override SpecialFunctions.besselj(n::Int32, x::Float64) = ccall("extern __nv_jn", llvmcall, Cdouble, (Int32, Cdouble), n, x)
@device_override SpecialFunctions.besselj(n::Int32, x::Float32) = ccall("extern __nv_jnf", llvmcall, Cfloat, (Int32, Cfloat), n, x)

@device_override SpecialFunctions.bessely0(x::Float64) = ccall("extern __nv_y0", llvmcall, Cdouble, (Cdouble,), x)
@device_override SpecialFunctions.bessely0(x::Float32) = ccall("extern __nv_y0f", llvmcall, Cfloat, (Cfloat,), x)

@device_override SpecialFunctions.bessely1(x::Float64) = ccall("extern __nv_y1", llvmcall, Cdouble, (Cdouble,), x)
@device_override SpecialFunctions.bessely1(x::Float32) = ccall("extern __nv_y1f", llvmcall, Cfloat, (Cfloat,), x)

@device_override SpecialFunctions.bessely(n::Int32, x::Float64) = ccall("extern __nv_yn", llvmcall, Cdouble, (Int32, Cdouble), n, x)
@device_override SpecialFunctions.bessely(n::Int32, x::Float32) = ccall("extern __nv_ynf", llvmcall, Cfloat, (Int32, Cfloat), n, x)


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

@device_override Base.fma(x::Float64, y::Float64, z::Float64) = ccall("extern __nv_fma", llvmcall, Cdouble, (Cdouble, Cdouble, Cdouble), x, y, z)
@device_override Base.fma(x::Float32, y::Float32, z::Float32) = ccall("extern __nv_fmaf", llvmcall, Cfloat, (Cfloat, Cfloat, Cfloat), x, y, z)
@device_override Base.fma(x::Float16, y::Float16, z::Float16) = ccall("llvm.fma.f16", llvmcall, Float16, (Float16, Float16, Float16), x, y, z)

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
