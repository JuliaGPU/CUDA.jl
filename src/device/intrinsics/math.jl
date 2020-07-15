# math functionality

## trigonometric

@inline cos(x::Float64) = ccall("extern __nv_cos", llvmcall, Cdouble, (Cdouble,), x)
@inline cos(x::Float32) = ccall("extern __nv_cosf", llvmcall, Cfloat, (Cfloat,), x)
@inline cos_fast(x::Float32) = ccall("extern __nv_fast_cosf", llvmcall, Cfloat, (Cfloat,), x)

@inline cospi(x::Float64) = ccall("extern __nv_cospi", llvmcall, Cdouble, (Cdouble,), x)
@inline cospi(x::Float32) = ccall("extern __nv_cospif", llvmcall, Cfloat, (Cfloat,), x)

@inline sin(x::Float64) = ccall("extern __nv_sin", llvmcall, Cdouble, (Cdouble,), x)
@inline sin(x::Float32) = ccall("extern __nv_sinf", llvmcall, Cfloat, (Cfloat,), x)
@inline sin_fast(x::Float32) = ccall("extern __nv_fast_sinf", llvmcall, Cfloat, (Cfloat,), x)

@inline sinpi(x::Float64) = ccall("extern __nv_sinpi", llvmcall, Cdouble, (Cdouble,), x)
@inline sinpi(x::Float32) = ccall("extern __nv_sinpif", llvmcall, Cfloat, (Cfloat,), x)

@inline tan(x::Float64) = ccall("extern __nv_tan", llvmcall, Cdouble, (Cdouble,), x)
@inline tan(x::Float32) = ccall("extern __nv_tanf", llvmcall, Cfloat, (Cfloat,), x)
@inline tan_fast(x::Float32) = ccall("extern __nv_fast_tanf", llvmcall, Cfloat, (Cfloat,), x)


## inverse trigonometric

@inline acos(x::Float64) = ccall("extern __nv_acos", llvmcall, Cdouble, (Cdouble,), x)
@inline acos(x::Float32) = ccall("extern __nv_acosf", llvmcall, Cfloat, (Cfloat,), x)

@inline asin(x::Float64) = ccall("extern __nv_asin", llvmcall, Cdouble, (Cdouble,), x)
@inline asin(x::Float32) = ccall("extern __nv_asinf", llvmcall, Cfloat, (Cfloat,), x)

@inline atan(x::Float64) = ccall("extern __nv_atan", llvmcall, Cdouble, (Cdouble,), x)
@inline atan(x::Float32) = ccall("extern __nv_atanf", llvmcall, Cfloat, (Cfloat,), x)

# ! atan2 is equivalent to Base.atan
@inline atan2(x::Float64, y::Float64) = ccall("extern __nv_atan2", llvmcall, Cdouble, (Cdouble, Cdouble), x, y)
@inline atan2(x::Float32, y::Float32) = ccall("extern __nv_atan2f", llvmcall, Cfloat, (Cfloat, Cfloat), x, y)
@inline atan(x::Float64, y::Float64) = atan2(x, y)
@inline atan(x::Float32, y::Float32) = atan2(x, y)

@inline angle(x::ComplexF64) = atan2(x.im, x.re)
@inline angle(x::ComplexF32) = atan2(x.im, x.re)
@inline angle(x::Float64) = signbit(x,) * 3.141592653589793
@inline angle(x::Float32) = signbit(x,) * 3.1415927f0

## hyperbolic

@inline cosh(x::Float64) = ccall("extern __nv_cosh", llvmcall, Cdouble, (Cdouble,), x)
@inline cosh(x::Float32) = ccall("extern __nv_coshf", llvmcall, Cfloat, (Cfloat,), x)

@inline sinh(x::Float64) = ccall("extern __nv_sinh", llvmcall, Cdouble, (Cdouble,), x)
@inline sinh(x::Float32) = ccall("extern __nv_sinhf", llvmcall, Cfloat, (Cfloat,), x)

@inline tanh(x::Float64) = ccall("extern __nv_tanh", llvmcall, Cdouble, (Cdouble,), x)
@inline tanh(x::Float32) = ccall("extern __nv_tanhf", llvmcall, Cfloat, (Cfloat,), x)


## inverse hyperbolic

@inline acosh(x::Float64) = ccall("extern __nv_acosh", llvmcall, Cdouble, (Cdouble,), x)
@inline acosh(x::Float32) = ccall("extern __nv_acoshf", llvmcall, Cfloat, (Cfloat,), x)

@inline asinh(x::Float64) = ccall("extern __nv_asinh", llvmcall, Cdouble, (Cdouble,), x)
@inline asinh(x::Float32) = ccall("extern __nv_asinhf", llvmcall, Cfloat, (Cfloat,), x)

@inline atanh(x::Float64) = ccall("extern __nv_atanh", llvmcall, Cdouble, (Cdouble,), x)
@inline atanh(x::Float32) = ccall("extern __nv_atanhf", llvmcall, Cfloat, (Cfloat,), x)


## logarithmic

@inline log(x::Float64) = ccall("extern __nv_log", llvmcall, Cdouble, (Cdouble,), x)
@inline log(x::Float32) = ccall("extern __nv_logf", llvmcall, Cfloat, (Cfloat,), x)
@inline log_fast(x::Float32) = ccall("extern __nv_fast_logf", llvmcall, Cfloat, (Cfloat,), x)

@inline log(x::ComplexF64) = log(abs(x,)) + im * angle(x,)
@inline log(x::ComplexF32) = log(abs(x,)) + im * angle(x,)
@inline log_fast(x::ComplexF32) = log_fast(abs(x,)) + im * angle(x,)

@inline log10(x::Float64) = ccall("extern __nv_log10", llvmcall, Cdouble, (Cdouble,), x)
@inline log10(x::Float32) = ccall("extern __nv_log10f", llvmcall, Cfloat, (Cfloat,), x)
@inline log10_fast(x::Float32) = ccall("extern __nv_fast_log10f", llvmcall, Cfloat, (Cfloat,), x)

@inline log1p(x::Float64) = ccall("extern __nv_log1p", llvmcall, Cdouble, (Cdouble,), x)
@inline log1p(x::Float32) = ccall("extern __nv_log1pf", llvmcall, Cfloat, (Cfloat,), x)

@inline log2(x::Float64) = ccall("extern __nv_log2", llvmcall, Cdouble, (Cdouble,), x)
@inline log2(x::Float32) = ccall("extern __nv_log2f", llvmcall, Cfloat, (Cfloat,), x)
@inline log2_fast(x::Float32) = ccall("extern __nv_fast_log2f", llvmcall, Cfloat, (Cfloat,), x)

@inline logb(x::Float64) = ccall("extern __nv_logb", llvmcall, Cdouble, (Cdouble,), x)
@inline logb(x::Float32) = ccall("extern __nv_logbf", llvmcall, Cfloat, (Cfloat,), x)

@inline ilogb(x::Float64) = ccall("extern __nv_ilogb", llvmcall, Int32, (Cdouble,), x)
@inline ilogb(x::Float32) = ccall("extern __nv_ilogbf", llvmcall, Int32, (Cfloat,), x)


## exponential

@inline exp(x::Float64) = ccall("extern __nv_exp", llvmcall, Cdouble, (Cdouble,), x)
@inline exp(x::Float32) = ccall("extern __nv_expf", llvmcall, Cfloat, (Cfloat,), x)
@inline exp_fast(x::Float32) = ccall("extern __nv_fast_expf", llvmcall, Cfloat, (Cfloat,), x)

@inline exp2(x::Float64) = ccall("extern __nv_exp2", llvmcall, Cdouble, (Cdouble,), x)
@inline exp2(x::Float32) = ccall("extern __nv_exp2f", llvmcall, Cfloat, (Cfloat,), x)

@inline exp10(x::Float64) = ccall("extern __nv_exp10", llvmcall, Cdouble, (Cdouble,), x)
@inline exp10(x::Float32) = ccall("extern __nv_exp10f", llvmcall, Cfloat, (Cfloat,), x)
@inline exp10_fast(x::Float32) = ccall("extern __nv_fast_exp10f", llvmcall, Cfloat, (Cfloat,), x)

@inline expm1(x::Float64) = ccall("extern __nv_expm1", llvmcall, Cdouble, (Cdouble,), x)
@inline expm1(x::Float32) = ccall("extern __nv_expm1f", llvmcall, Cfloat, (Cfloat,), x)

@inline ldexp(x::Float64, y::Int32) = ccall("extern __nv_ldexp", llvmcall, Cdouble, (Cdouble, Int32), x, y)
@inline ldexp(x::Float32, y::Int32) = ccall("extern __nv_ldexpf", llvmcall, Cfloat, (Cfloat, Int32), x, y)

@inline exp(x::Complex{Float64}) = exp(x.re) * (cos(x.im) + 1.0im * sin(x.im))
@inline exp(x::Complex{Float32}) = exp(x.re) * (cos(x.im) + 1.0f0im * sin(x.im))
@inline exp_fast(x::Complex{Float32}) = exp_fast(x.re) * (cos_fast(x.im) + 1.0f0im * sin_fast(x.im))

## error

@inline erf(x::Float64) = ccall("extern __nv_erf", llvmcall, Cdouble, (Cdouble,), x)
@inline erf(x::Float32) = ccall("extern __nv_erff", llvmcall, Cfloat, (Cfloat,), x)

@inline erfinv(x::Float64) = ccall("extern __nv_erfinv", llvmcall, Cdouble, (Cdouble,), x)
@inline erfinv(x::Float32) = ccall("extern __nv_erfinvf", llvmcall, Cfloat, (Cfloat,), x)

@inline erfc(x::Float64) = ccall("extern __nv_erfc", llvmcall, Cdouble, (Cdouble,), x)
@inline erfc(x::Float32) = ccall("extern __nv_erfcf", llvmcall, Cfloat, (Cfloat,), x)

@inline erfcinv(x::Float64) = ccall("extern __nv_erfcinv", llvmcall, Cdouble, (Cdouble,), x)
@inline erfcinv(x::Float32) = ccall("extern __nv_erfcinvf", llvmcall, Cfloat, (Cfloat,), x)

@inline erfcx(x::Float64) = ccall("extern __nv_erfcx", llvmcall, Cdouble, (Cdouble,), x)
@inline erfcx(x::Float32) = ccall("extern __nv_erfcxf", llvmcall, Cfloat, (Cfloat,), x)


## integer handling (bit twiddling)

@inline brev(x::Int32) =   ccall("extern __nv_brev", llvmcall, Int32, (Int32,), x)
@inline brev(x::Int64) =   ccall("extern __nv_brevll", llvmcall, Int64, (Int64,), x)

@inline clz(x::Int32) =   ccall("extern __nv_clz", llvmcall, Int32, (Int32,), x)
@inline clz(x::Int64) =   ccall("extern __nv_clzll", llvmcall, Int32, (Int64,), x)

@inline ffs(x::Int32) = ccall("extern __nv_ffs", llvmcall, Int32, (Int32,), x)
@inline ffs(x::Int64) = ccall("extern __nv_ffsll", llvmcall, Int32, (Int64,), x)

@inline byte_perm(x::Int32, y::Int32, z::Int32) = ccall("extern __nv_byte_perm", llvmcall, Int32, (Int32, Int32, Int32), x, y, z)

@inline popc(x::Int32) = ccall("extern __nv_popc", llvmcall, Int32, (Int32,), x)
@inline popc(x::Int64) = ccall("extern __nv_popcll", llvmcall, Int32, (Int64,), x)


## floating-point handling

@inline isfinite(x::Float32) = (ccall("extern __nv_finitef", llvmcall, Int32, (Cfloat,), x)) != 0
@inline isfinite(x::Float64) = (ccall("extern __nv_isfinited", llvmcall, Int32, (Cdouble,), x)) != 0

@inline isinf(x::Float64) = (ccall("extern __nv_isinfd", llvmcall, Int32, (Cdouble,), x)) != 0
@inline isinf(x::Float32) = (ccall("extern __nv_isinff", llvmcall, Int32, (Cfloat,), x)) != 0

@inline isnan(x::Float64) = (ccall("extern __nv_isnand", llvmcall, Int32, (Cdouble,), x)) != 0
@inline isnan(x::Float32) = (ccall("extern __nv_isnanf", llvmcall, Int32, (Cfloat,), x)) != 0

@inline nearbyint(x::Float64) = ccall("extern __nv_nearbyint", llvmcall, Cdouble, (Cdouble,), x)
@inline nearbyint(x::Float32) = ccall("extern __nv_nearbyintf", llvmcall, Cfloat, (Cfloat,), x)

@inline nextafter(x::Float64, y::Float64) = ccall("extern __nv_nextafter", llvmcall, Cdouble, (Cdouble, Cdouble), x, y)
@inline nextafter(x::Float32, y::Float32) = ccall("extern __nv_nextafterf", llvmcall, Cfloat, (Cfloat, Cfloat), x, y)


## sign handling

@inline signbit(x::Float64) = (ccall("extern __nv_signbitd", llvmcall, Int32, (Cdouble,), x)) != 0
@inline signbit(x::Float32) = (ccall("extern __nv_signbitf", llvmcall, Int32, (Cfloat,), x)) != 0

@inline copysign(x::Float64, y::Float64) = ccall("extern __nv_copysign", llvmcall, Cdouble, (Cdouble, Cdouble), x, y)
@inline copysign(x::Float32, y::Float32) = ccall("extern __nv_copysignf", llvmcall, Cfloat, (Cfloat, Cfloat), x, y)

@inline abs(x::Int32) =   ccall("extern __nv_abs", llvmcall, Int32, (Int32,), x)
@inline abs(f::Float64) = ccall("extern __nv_fabs", llvmcall, Cdouble, (Cdouble,), f)
@inline abs(f::Float32) = ccall("extern __nv_fabsf", llvmcall, Cfloat, (Cfloat,), f)
@inline abs(x::Int64) =   ccall("extern __nv_llabs", llvmcall, Int64, (Int64,), x)

@inline abs(x::Complex{Float64}) = hypot(x.re, x.im)
@inline abs(x::Complex{Float32}) = hypot(x.re, x.im)

## roots and powers

@inline sqrt(x::Float64) = ccall("extern __nv_sqrt", llvmcall, Cdouble, (Cdouble,), x)
@inline sqrt(x::Float32) = ccall("extern __nv_sqrtf", llvmcall, Cfloat, (Cfloat,), x)

@inline rsqrt(x::Float64) = ccall("extern __nv_rsqrt", llvmcall, Cdouble, (Cdouble,), x)
@inline rsqrt(x::Float32) = ccall("extern __nv_rsqrtf", llvmcall, Cfloat, (Cfloat,), x)

@inline cbrt(x::Float64) = ccall("extern __nv_cbrt", llvmcall, Cdouble, (Cdouble,), x)
@inline cbrt(x::Float32) = ccall("extern __nv_cbrtf", llvmcall, Cfloat, (Cfloat,), x)

@inline rcbrt(x::Float64) = ccall("extern __nv_rcbrt", llvmcall, Cdouble, (Cdouble,), x)
@inline rcbrt(x::Float32) = ccall("extern __nv_rcbrtf", llvmcall, Cfloat, (Cfloat,), x)

@inline pow(x::Float64, y::Float64) = ccall("extern __nv_pow", llvmcall, Cdouble, (Cdouble, Cdouble), x, y)
@inline pow(x::Float32, y::Float32) = ccall("extern __nv_powf", llvmcall, Cfloat, (Cfloat, Cfloat), x, y)
@inline pow_fast(x::Float32, y::Float32) = ccall("extern __nv_fast_powf", llvmcall, Cfloat, (Cfloat, Cfloat), x, y)
@inline pow(x::Float64, y::Int32) =   ccall("extern __nv_powi", llvmcall, Cdouble, (Cdouble, Int32), x, y)
@inline pow(x::Float32, y::Int32) =   ccall("extern __nv_powif", llvmcall, Cfloat, (Cfloat, Int32), x, y)
@inline pow(x::Union{Float32, Float64}, y::Int64) = pow(x, Int32(y,))

@inline abs2(x::Complex{Float64}) = x.re * x.re + x.im * x.im
@inline abs2(x::Complex{Float32}) = x.re * x.re + x.im * x.im

## rounding and selection

# TODO: differentiate in return type, map correctly
# @inline round(x::Float64) = ccall("extern __nv_llround", llvmcall, Int64, (Cdouble,), x)
# @inline round(x::Float32) = ccall("extern __nv_llroundf", llvmcall, Int64, (Cfloat,), x)
# @inline round(x::Float64) = ccall("extern __nv_round", llvmcall, Cdouble, (Cdouble,), x)
# @inline round(x::Float32) = ccall("extern __nv_roundf", llvmcall, Cfloat, (Cfloat,), x)

# TODO: differentiate in return type, map correctly
# @inline rint(x::Float64) = ccall("extern __nv_llrint", llvmcall, Int64, (Cdouble,), x)
# @inline rint(x::Float32) = ccall("extern __nv_llrintf", llvmcall, Int64, (Cfloat,), x)
# @inline rint(x::Float64) = ccall("extern __nv_rint", llvmcall, Cdouble, (Cdouble,), x)
# @inline rint(x::Float32) = ccall("extern __nv_rintf", llvmcall, Cfloat, (Cfloat,), x)

# TODO: would conflict with trunc usage in this module
# @inline trunc(x::Float64) = ccall("extern __nv_trunc", llvmcall, Cdouble, (Cdouble,), x)
# @inline trunc(x::Float32) = ccall("extern __nv_truncf", llvmcall, Cfloat, (Cfloat,), x)

@inline ceil(x::Float64) = ccall("extern __nv_ceil", llvmcall, Cdouble, (Cdouble,), x)
@inline ceil(x::Float32) = ccall("extern __nv_ceilf", llvmcall, Cfloat, (Cfloat,), x)

@inline floor(f::Float64) = ccall("extern __nv_floor", llvmcall, Cdouble, (Cdouble,), f)
@inline floor(f::Float32) = ccall("extern __nv_floorf", llvmcall, Cfloat, (Cfloat,), f)

@inline min(x::Int32, y::Int32) = ccall("extern __nv_min", llvmcall, Int32, (Int32, Int32), x, y)
@inline min(x::Int64, y::Int64) = ccall("extern __nv_llmin", llvmcall, Int64, (Int64, Int64), x, y)
@inline min(x::UInt32, y::UInt32) = convert(UInt32, ccall("extern __nv_umin", llvmcall, Int32, (Int32, Int32), x, y))
@inline min(x::UInt64, y::UInt64) = convert(UInt64, ccall("extern __nv_ullmin", llvmcall, Int64, (Int64, Int64), x, y))
@inline min(x::Float64, y::Float64) = ccall("extern __nv_fmin", llvmcall, Cdouble, (Cdouble, Cdouble), x, y)
@inline min(x::Float32, y::Float32) = ccall("extern __nv_fminf", llvmcall, Cfloat, (Cfloat, Cfloat), x, y)

@inline max(x::Int32, y::Int32) = ccall("extern __nv_max", llvmcall, Int32, (Int32, Int32), x, y)
@inline max(x::Int64, y::Int64) = ccall("extern __nv_llmax", llvmcall, Int64, (Int64, Int64), x, y)
@inline max(x::UInt32, y::UInt32) = convert(UInt32, ccall("extern __nv_umax", llvmcall, Int32, (Int32, Int32), x, y))
@inline max(x::UInt64, y::UInt64) = convert(UInt64, ccall("extern __nv_ullmax", llvmcall, Int64, (Int64, Int64), x, y))
@inline max(x::Float64, y::Float64) = ccall("extern __nv_fmax", llvmcall, Cdouble, (Cdouble, Cdouble), x, y)
@inline max(x::Float32, y::Float32) = ccall("extern __nv_fmaxf", llvmcall, Cfloat, (Cfloat, Cfloat), x, y)

@inline saturate(x::Float32) = ccall("extern __nv_saturatef", llvmcall, Cfloat, (Cfloat,), x)


## division and remainder

@inline mod(x::Float64, y::Float64) = ccall("extern __nv_fmod", llvmcall, Cdouble, (Cdouble, Cdouble), x, y)
@inline mod(x::Float32, y::Float32) = ccall("extern __nv_fmodf", llvmcall, Cfloat, (Cfloat, Cfloat), x, y)

@inline rem(x::Float64, y::Float64) = ccall("extern __nv_remainder", llvmcall, Cdouble, (Cdouble, Cdouble), x, y)
@inline rem(x::Float32, y::Float32) = ccall("extern __nv_remainderf", llvmcall, Cfloat, (Cfloat, Cfloat), x, y)

@inline div_fast(x::Float32, y::Float32) = ccall("extern __nv_fast_fdividef", llvmcall, Cfloat, (Cfloat, Cfloat), x, y)


## gamma function

@inline lgamma(x::Float64) = ccall("extern __nv_lgamma", llvmcall, Cdouble, (Cdouble,), x)
@inline lgamma(x::Float32) = ccall("extern __nv_lgammaf", llvmcall, Cfloat, (Cfloat,), x)

@inline tgamma(x::Float64) = ccall("extern __nv_tgamma", llvmcall, Cdouble, (Cdouble,), x)
@inline tgamma(x::Float32) = ccall("extern __nv_tgammaf", llvmcall, Cfloat, (Cfloat,), x)


## Bessel

@inline j0(x::Float64) = ccall("extern __nv_j0", llvmcall, Cdouble, (Cdouble,), x)
@inline j0(x::Float32) = ccall("extern __nv_j0f", llvmcall, Cfloat, (Cfloat,), x)

@inline j1(x::Float64) = ccall("extern __nv_j1", llvmcall, Cdouble, (Cdouble,), x)
@inline j1(x::Float32) = ccall("extern __nv_j1f", llvmcall, Cfloat, (Cfloat,), x)

@inline jn(n::Int32, x::Float64) = ccall("extern __nv_jn", llvmcall, Cdouble, (Int32, Cdouble), n, x)
@inline jn(n::Int32, x::Float32) = ccall("extern __nv_jnf", llvmcall, Cfloat, (Int32, Cfloat), n, x)

@inline y0(x::Float64) = ccall("extern __nv_y0", llvmcall, Cdouble, (Cdouble,), x)
@inline y0(x::Float32) = ccall("extern __nv_y0f", llvmcall, Cfloat, (Cfloat,), x)

@inline y1(x::Float64) = ccall("extern __nv_y1", llvmcall, Cdouble, (Cdouble,), x)
@inline y1(x::Float32) = ccall("extern __nv_y1f", llvmcall, Cfloat, (Cfloat,), x)

@inline yn(n::Int32, x::Float64) = ccall("extern __nv_yn", llvmcall, Cdouble, (Int32, Cdouble), n, x)
@inline yn(n::Int32, x::Float32) = ccall("extern __nv_ynf", llvmcall, Cfloat, (Int32, Cfloat), n, x)


## distributions

@inline normcdf(x::Float64) = ccall("extern __nv_normcdf", llvmcall, Cdouble, (Cdouble,), x)
@inline normcdf(x::Float32) = ccall("extern __nv_normcdff", llvmcall, Cfloat, (Cfloat,), x)

@inline normcdfinv(x::Float64) = ccall("extern __nv_normcdfinv", llvmcall, Cdouble, (Cdouble,), x)
@inline normcdfinv(x::Float32) = ccall("extern __nv_normcdfinvf", llvmcall, Cfloat, (Cfloat,), x)



#
# Unsorted
#

@inline hypot(x::Float64, y::Float64) = ccall("extern __nv_hypot", llvmcall, Cdouble, (Cdouble, Cdouble), x, y)
@inline hypot(x::Float32, y::Float32) = ccall("extern __nv_hypotf", llvmcall, Cfloat, (Cfloat, Cfloat), x, y)

@inline fma(x::Float64, y::Float64, z::Float64) = ccall("extern __nv_fma", llvmcall, Cdouble, (Cdouble, Cdouble, Cdouble), x, y, z)
@inline fma(x::Float32, y::Float32, z::Float32) = ccall("extern __nv_fmaf", llvmcall, Cfloat, (Cfloat, Cfloat, Cfloat), x, y, z)

@inline sad(x::Int32, y::Int32, z::Int32) = ccall("extern __nv_sad", llvmcall, Int32, (Int32, Int32, Int32), x, y, z)
@inline sad(x::UInt32, y::UInt32, z::UInt32) = convert(UInt32, ccall("extern __nv_usad", llvmcall, Int32, (Int32, Int32, Int32), x, y, z))

@inline dim(x::Float64, y::Float64) = ccall("extern __nv_fdim", llvmcall, Cdouble, (Cdouble, Cdouble), x, y)
@inline dim(x::Float32, y::Float32) = ccall("extern __nv_fdimf", llvmcall, Cfloat, (Cfloat, Cfloat), x, y)

@inline mul24(x::Int32, y::Int32) = ccall("extern __nv_mul24", llvmcall, Int32, (Int32, Int32), x, y)
@inline mul24(x::UInt32, y::UInt32) = convert(UInt32, ccall("extern __nv_umul24", llvmcall, Int32, (Int32, Int32), x, y))

@inline mul64hi(x::Int64, y::Int64) = ccall("extern __nv_mul64hi", llvmcall, Int64, (Int64, Int64), x, y)
@inline mul64hi(x::UInt64, y::UInt64) = convert(UInt64, ccall("extern __nv_umul64hi", llvmcall, Int64, (Int64, Int64), x, y))
@inline mulhi(x::Int32, y::Int32) = ccall("extern __nv_mulhi", llvmcall, Int32, (Int32, Int32), x, y)
@inline mulhi(x::UInt32, y::UInt32) = convert(UInt32, ccall("extern __nv_umulhi", llvmcall, Int32, (Int32, Int32), x, y))

@inline hadd(x::Int32, y::Int32) = ccall("extern __nv_hadd", llvmcall, Int32, (Int32, Int32), x, y)
@inline hadd(x::UInt32, y::UInt32) = convert(UInt32, ccall("extern __nv_uhadd", llvmcall, Int32, (Int32, Int32), x, y))

@inline rhadd(x::Int32, y::Int32) = ccall("extern __nv_rhadd", llvmcall, Int32, (Int32, Int32), x, y)
@inline rhadd(x::UInt32, y::UInt32) = convert(UInt32, ccall("extern __nv_urhadd", llvmcall, Int32, (Int32, Int32), x, y))

@inline scalbn(x::Float64, y::Int32) = ccall("extern __nv_scalbn", llvmcall, Cdouble, (Cdouble, Int32), x, y)
@inline scalbn(x::Float32, y::Int32) = ccall("extern __nv_scalbnf", llvmcall, Cfloat, (Cfloat, Int32), x, y)
