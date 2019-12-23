# math functionality

## trigonometric

@inline cos(x::Float64) = @wrap __nv_cos(x::double)::double
@inline cos(x::Float32) = @wrap __nv_cosf(x::float)::float
@inline cos_fast(x::Float32) = @wrap __nv_fast_cosf(x::float)::float

@inline cospi(x::Float64) = @wrap __nv_cospi(x::double)::double
@inline cospi(x::Float32) = @wrap __nv_cospif(x::float)::float

@inline sin(x::Float64) = @wrap __nv_sin(x::double)::double
@inline sin(x::Float32) = @wrap __nv_sinf(x::float)::float
@inline sin_fast(x::Float32) = @wrap __nv_fast_sinf(x::float)::float

@inline sinpi(x::Float64) = @wrap __nv_sinpi(x::double)::double
@inline sinpi(x::Float32) = @wrap __nv_sinpif(x::float)::float

@inline tan(x::Float64) = @wrap __nv_tan(x::double)::double
@inline tan(x::Float32) = @wrap __nv_tanf(x::float)::float
@inline tan_fast(x::Float32) = @wrap __nv_fast_tanf(x::float)::float


## inverse trigonometric

@inline acos(x::Float64) = @wrap __nv_acos(x::double)::double
@inline acos(x::Float32) = @wrap __nv_acosf(x::float)::float

@inline asin(x::Float64) = @wrap __nv_asin(x::double)::double
@inline asin(x::Float32) = @wrap __nv_asinf(x::float)::float

@inline atan(x::Float64) = @wrap __nv_atan(x::double)::double
@inline atan(x::Float32) = @wrap __nv_atanf(x::float)::float

# ! CUDAnative.atan2 is equivalent to Base.atan
@inline atan2(x::Float64, y::Float64) = @wrap __nv_atan2(x::double, y::double)::double
@inline atan2(x::Float32, y::Float32) = @wrap __nv_atan2f(x::float, y::float)::float

@inline angle(x::ComplexF64) = atan2(x.im, x.re)
@inline angle(x::ComplexF32) = atan2(x.im, x.re)
@inline angle(x::Float64) = signbit(x) * 3.141592653589793
@inline angle(x::Float32) = signbit(x) * 3.1415927f0

## hyperbolic

@inline cosh(x::Float64) = @wrap __nv_cosh(x::double)::double
@inline cosh(x::Float32) = @wrap __nv_coshf(x::float)::float

@inline sinh(x::Float64) = @wrap __nv_sinh(x::double)::double
@inline sinh(x::Float32) = @wrap __nv_sinhf(x::float)::float

@inline tanh(x::Float64) = @wrap __nv_tanh(x::double)::double
@inline tanh(x::Float32) = @wrap __nv_tanhf(x::float)::float


## inverse hyperbolic

@inline acosh(x::Float64) = @wrap __nv_acosh(x::double)::double
@inline acosh(x::Float32) = @wrap __nv_acoshf(x::float)::float

@inline asinh(x::Float64) = @wrap __nv_asinh(x::double)::double
@inline asinh(x::Float32) = @wrap __nv_asinhf(x::float)::float

@inline atanh(x::Float64) = @wrap __nv_atanh(x::double)::double
@inline atanh(x::Float32) = @wrap __nv_atanhf(x::float)::float


## logarithmic

@inline log(x::Float64) = @wrap __nv_log(x::double)::double
@inline log(x::Float32) = @wrap __nv_logf(x::float)::float
@inline log_fast(x::Float32) = @wrap __nv_fast_logf(x::float)::float

@inline log(x::ComplexF64) = log(abs(x)) + im * angle(x)
@inline log(x::ComplexF32) = log(abs(x)) + im * angle(x)
@inline log_fast(x::ComplexF32) = log_fast(abs(x)) + im * angle(x)

@inline log10(x::Float64) = @wrap __nv_log10(x::double)::double
@inline log10(x::Float32) = @wrap __nv_log10f(x::float)::float
@inline log10_fast(x::Float32) = @wrap __nv_fast_log10f(x::float)::float

@inline log1p(x::Float64) = @wrap __nv_log1p(x::double)::double
@inline log1p(x::Float32) = @wrap __nv_log1pf(x::float)::float

@inline log2(x::Float64) = @wrap __nv_log2(x::double)::double
@inline log2(x::Float32) = @wrap __nv_log2f(x::float)::float
@inline log2_fast(x::Float32) = @wrap __nv_fast_log2f(x::float)::float

@inline logb(x::Float64) = @wrap __nv_logb(x::double)::double
@inline logb(x::Float32) = @wrap __nv_logbf(x::float)::float

@inline ilogb(x::Float64) = @wrap __nv_ilogb(x::double)::i32
@inline ilogb(x::Float32) = @wrap __nv_ilogbf(x::float)::i32


## exponential

@inline exp(x::Float64) = @wrap __nv_exp(x::double)::double
@inline exp(x::Float32) = @wrap __nv_expf(x::float)::float
@inline exp_fast(x::Float32) = @wrap __nv_fast_expf(x::float)::float

@inline exp2(x::Float64) = @wrap __nv_exp2(x::double)::double
@inline exp2(x::Float32) = @wrap __nv_exp2f(x::float)::float

@inline exp10(x::Float64) = @wrap __nv_exp10(x::double)::double
@inline exp10(x::Float32) = @wrap __nv_exp10f(x::float)::float
@inline exp10_fast(x::Float32) = @wrap __nv_fast_exp10f(x::float)::float

@inline expm1(x::Float64) = @wrap __nv_expm1(x::double)::double
@inline expm1(x::Float32) = @wrap __nv_expm1f(x::float)::float

@inline ldexp(x::Float64, y::Int32) = @wrap __nv_ldexp(x::double, y::i32)::double
@inline ldexp(x::Float32, y::Int32) = @wrap __nv_ldexpf(x::float, y::i32)::float

@inline exp(x::Complex{Float64}) = exp(x.re) * (cos(x.im) + 1.0im * sin(x.im))
@inline exp(x::Complex{Float32}) = exp(x.re) * (cos(x.im) + 1.0f0im * sin(x.im))
@inline exp_fast(x::Complex{Float32}) = exp_fast(x.re) * (cos_fast(x.im) + 1.0f0im * sin_fast(x.im))

## error

@inline erf(x::Float64) = @wrap __nv_erf(x::double)::double
@inline erf(x::Float32) = @wrap __nv_erff(x::float)::float

@inline erfinv(x::Float64) = @wrap __nv_erfinv(x::double)::double
@inline erfinv(x::Float32) = @wrap __nv_erfinvf(x::float)::float

@inline erfc(x::Float64) = @wrap __nv_erfc(x::double)::double
@inline erfc(x::Float32) = @wrap __nv_erfcf(x::float)::float

@inline erfcinv(x::Float64) = @wrap __nv_erfcinv(x::double)::double
@inline erfcinv(x::Float32) = @wrap __nv_erfcinvf(x::float)::float

@inline erfcx(x::Float64) = @wrap __nv_erfcx(x::double)::double
@inline erfcx(x::Float32) = @wrap __nv_erfcxf(x::float)::float


## integer handling (bit twiddling)

@inline brev(x::Int32) =   @wrap __nv_brev(x::i32)::i32
@inline brev(x::Int64) =   @wrap __nv_brevll(x::i64)::i64

@inline clz(x::Int32) =   @wrap __nv_clz(x::i32)::i32
@inline clz(x::Int64) =   @wrap __nv_clzll(x::i64)::i32

@inline ffs(x::Int32) = @wrap __nv_ffs(x::i32)::i32
@inline ffs(x::Int64) = @wrap __nv_ffsll(x::i64)::i32

@inline byte_perm(x::Int32, y::Int32, z::Int32) = @wrap __nv_byte_perm(x::i32, y::i32, z::i32)::i32

@inline popc(x::Int32) = @wrap __nv_popc(x::i32)::i32
@inline popc(x::Int64) = @wrap __nv_popcll(x::i64)::i32


## floating-point handling

@inline isfinite(x::Float32) = (@wrap __nv_finitef(x::float)::i32) != 0
@inline isfinite(x::Float64) = (@wrap __nv_isfinited(x::double)::i32) != 0

@inline isinf(x::Float64) = (@wrap __nv_isinfd(x::double)::i32) != 0
@inline isinf(x::Float32) = (@wrap __nv_isinff(x::float)::i32) != 0

@inline isnan(x::Float64) = (@wrap __nv_isnand(x::double)::i32) != 0
@inline isnan(x::Float32) = (@wrap __nv_isnanf(x::float)::i32) != 0

@inline nearbyint(x::Float64) = @wrap __nv_nearbyint(x::double)::double
@inline nearbyint(x::Float32) = @wrap __nv_nearbyintf(x::float)::float

@inline nextafter(x::Float64, y::Float64) = @wrap __nv_nextafter(x::double, y::double)::double
@inline nextafter(x::Float32, y::Float32) = @wrap __nv_nextafterf(x::float, y::float)::float


## sign handling

@inline signbit(x::Float64) = (@wrap __nv_signbitd(x::double)::i32) != 0
@inline signbit(x::Float32) = (@wrap __nv_signbitf(x::float)::i32) != 0

@inline copysign(x::Float64, y::Float64) = @wrap __nv_copysign(x::double, y::double)::double
@inline copysign(x::Float32, y::Float32) = @wrap __nv_copysignf(x::float, y::float)::float

@inline abs(x::Int32) =   @wrap __nv_abs(x::i32)::i32
@inline abs(f::Float64) = @wrap __nv_fabs(f::double)::double
@inline abs(f::Float32) = @wrap __nv_fabsf(f::float)::float
@inline abs(x::Int64) =   @wrap __nv_llabs(x::i64)::i64

@inline abs(x::Complex{Float64}) = hypot(x.re, x.im)
@inline abs(x::Complex{Float32}) = hypot(x.re, x.im)

## roots and powers

@inline sqrt(x::Float64) = @wrap __nv_sqrt(x::double)::double
@inline sqrt(x::Float32) = @wrap __nv_sqrtf(x::float)::float

@inline rsqrt(x::Float64) = @wrap __nv_rsqrt(x::double)::double
@inline rsqrt(x::Float32) = @wrap __nv_rsqrtf(x::float)::float

@inline cbrt(x::Float64) = @wrap __nv_cbrt(x::double)::double
@inline cbrt(x::Float32) = @wrap __nv_cbrtf(x::float)::float

@inline rcbrt(x::Float64) = @wrap __nv_rcbrt(x::double)::double
@inline rcbrt(x::Float32) = @wrap __nv_rcbrtf(x::float)::float

@inline pow(x::Float64, y::Float64) = @wrap __nv_pow(x::double, y::double)::double
@inline pow(x::Float32, y::Float32) = @wrap __nv_powf(x::float, y::float)::float
@inline pow_fast(x::Float32, y::Float32) = @wrap __nv_fast_powf(x::float, y::float)::float
@inline pow(x::Float64, y::Int32) =   @wrap __nv_powi(x::double, y::i32)::double
@inline pow(x::Float32, y::Int32) =   @wrap __nv_powif(x::float, y::i32)::float
@inline pow(x::Union{Float32, Float64}, y::Int64) = pow(x, Int32(y))

@inline abs2(x::Complex{Float64}) = x.re * x.re + x.im * x.im
@inline abs2(x::Complex{Float32}) = x.re * x.re + x.im * x.im

## rounding and selection

# TODO: differentiate in return type, map correctly
# @inline round(x::Float64) = @wrap __nv_llround(x::double)::i64
# @inline round(x::Float32) = @wrap __nv_llroundf(x::float)::i64
# @inline round(x::Float64) = @wrap __nv_round(x::double)::double
# @inline round(x::Float32) = @wrap __nv_roundf(x::float)::float

# TODO: differentiate in return type, map correctly
# @inline rint(x::Float64) = @wrap __nv_llrint(x::double)::i64
# @inline rint(x::Float32) = @wrap __nv_llrintf(x::float)::i64
# @inline rint(x::Float64) = @wrap __nv_rint(x::double)::double
# @inline rint(x::Float32) = @wrap __nv_rintf(x::float)::float

# TODO: would conflict with trunc usage in this module
# @inline trunc(x::Float64) = @wrap __nv_trunc(x::double)::double
# @inline trunc(x::Float32) = @wrap __nv_truncf(x::float)::float

@inline ceil(x::Float64) = @wrap __nv_ceil(x::double)::double
@inline ceil(x::Float32) = @wrap __nv_ceilf(x::float)::float

@inline floor(f::Float64) = @wrap __nv_floor(f::double)::double
@inline floor(f::Float32) = @wrap __nv_floorf(f::float)::float

@inline min(x::Int32, y::Int32) = @wrap __nv_min(x::i32, y::i32)::i32
@inline min(x::Int64, y::Int64) = @wrap __nv_llmin(x::i64, y::i64)::i64
@inline min(x::UInt32, y::UInt32) = convert(UInt32, @wrap __nv_umin(x::i32, y::i32)::i32)
@inline min(x::UInt64, y::UInt64) = convert(UInt64, @wrap __nv_ullmin(x::i64, y::i64)::i64)
@inline min(x::Float64, y::Float64) = @wrap __nv_fmin(x::double, y::double)::double
@inline min(x::Float32, y::Float32) = @wrap __nv_fminf(x::float, y::float)::float

@inline max(x::Int32, y::Int32) = @wrap __nv_max(x::i32, y::i32)::i32
@inline max(x::Int64, y::Int64) = @wrap __nv_llmax(x::i64, y::i64)::i64
@inline max(x::UInt32, y::UInt32) = convert(UInt32, @wrap __nv_umax(x::i32, y::i32)::i32)
@inline max(x::UInt64, y::UInt64) = convert(UInt64, @wrap __nv_ullmax(x::i64, y::i64)::i64)
@inline max(x::Float64, y::Float64) = @wrap __nv_fmax(x::double, y::double)::double
@inline max(x::Float32, y::Float32) = @wrap __nv_fmaxf(x::float, y::float)::float

@inline saturate(x::Float32) = @wrap __nv_saturatef(x::float)::float


## division and remainder

@inline mod(x::Float64, y::Float64) = @wrap __nv_fmod(x::double, y::double)::double
@inline mod(x::Float32, y::Float32) = @wrap __nv_fmodf(x::float, y::float)::float

@inline rem(x::Float64, y::Float64) = @wrap __nv_remainder(x::double, y::double)::double
@inline rem(x::Float32, y::Float32) = @wrap __nv_remainderf(x::float, y::float)::float

@inline div_fast(x::Float32, y::Float32) = @wrap __nv_fast_fdividef(x::float, y::float)::float


## gamma function

@inline lgamma(x::Float64) = @wrap __nv_lgamma(x::double)::double
@inline lgamma(x::Float32) = @wrap __nv_lgammaf(x::float)::float

@inline tgamma(x::Float64) = @wrap __nv_tgamma(x::double)::double
@inline tgamma(x::Float32) = @wrap __nv_tgammaf(x::float)::float


## Bessel

@inline j0(x::Float64) = @wrap __nv_j0(x::double)::double
@inline j0(x::Float32) = @wrap __nv_j0f(x::float)::float

@inline j1(x::Float64) = @wrap __nv_j1(x::double)::double
@inline j1(x::Float32) = @wrap __nv_j1f(x::float)::float

@inline jn(n::Int32, x::Float64) = @wrap __nv_jn(n::i32, x::double)::double
@inline jn(n::Int32, x::Float32) = @wrap __nv_jnf(n::i32, x::float)::float

@inline y0(x::Float64) = @wrap __nv_y0(x::double)::double
@inline y0(x::Float32) = @wrap __nv_y0f(x::float)::float

@inline y1(x::Float64) = @wrap __nv_y1(x::double)::double
@inline y1(x::Float32) = @wrap __nv_y1f(x::float)::float

@inline yn(n::Int32, x::Float64) = @wrap __nv_yn(n::i32, x::double)::double
@inline yn(n::Int32, x::Float32) = @wrap __nv_ynf(n::i32, x::float)::float


## distributions

@inline normcdf(x::Float64) = @wrap __nv_normcdf(x::double)::double
@inline normcdf(x::Float32) = @wrap __nv_normcdff(x::float)::float

@inline normcdfinv(x::Float64) = @wrap __nv_normcdfinv(x::double)::double
@inline normcdfinv(x::Float32) = @wrap __nv_normcdfinvf(x::float)::float



#
# Unsorted
#

@inline hypot(x::Float64, y::Float64) = @wrap __nv_hypot(x::double, y::double)::double
@inline hypot(x::Float32, y::Float32) = @wrap __nv_hypotf(x::float, y::float)::float

@inline fma(x::Float64, y::Float64, z::Float64) = @wrap __nv_fma(x::double, y::double, z::double)::double
@inline fma(x::Float32, y::Float32, z::Float32) = @wrap __nv_fmaf(x::float, y::float, z::float)::float

@inline sad(x::Int32, y::Int32, z::Int32) = @wrap __nv_sad(x::i32, y::i32, z::i32)::i32
@inline sad(x::UInt32, y::UInt32, z::UInt32) = convert(UInt32, @wrap __nv_usad(x::i32, y::i32, z::i32)::i32)

@inline dim(x::Float64, y::Float64) = @wrap __nv_fdim(x::double, y::double)::double
@inline dim(x::Float32, y::Float32) = @wrap __nv_fdimf(x::float, y::float)::float

@inline mul24(x::Int32, y::Int32) = @wrap __nv_mul24(x::i32, y::i32)::i32
@inline mul24(x::UInt32, y::UInt32) = convert(UInt32, @wrap __nv_umul24(x::i32, y::i32)::i32)

@inline mul64hi(x::Int64, y::Int64) = @wrap __nv_mul64hi(x::i64, y::i64)::i64
@inline mul64hi(x::UInt64, y::UInt64) = convert(UInt64, @wrap __nv_umul64hi(x::i64, y::i64)::i64)
@inline mulhi(x::Int32, y::Int32) = @wrap __nv_mulhi(x::i32, y::i32)::i32
@inline mulhi(x::UInt32, y::UInt32) = convert(UInt32, @wrap __nv_umulhi(x::i32, y::i32)::i32)

@inline hadd(x::Int32, y::Int32) = @wrap __nv_hadd(x::i32, y::i32)::i32
@inline hadd(x::UInt32, y::UInt32) = convert(UInt32, @wrap __nv_uhadd(x::i32, y::i32)::i32)

@inline rhadd(x::Int32, y::Int32) = @wrap __nv_rhadd(x::i32, y::i32)::i32
@inline rhadd(x::UInt32, y::UInt32) = convert(UInt32, @wrap __nv_urhadd(x::i32, y::i32)::i32)

@inline scalbn(x::Float64, y::Int32) = @wrap __nv_scalbn(x::double, y::i32)::double
@inline scalbn(x::Float32, y::Int32) = @wrap __nv_scalbnf(x::float, y::i32)::float
