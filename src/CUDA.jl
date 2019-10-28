module CUDA


using Reexport


@reexport using CUDAdrv

@eval $(Symbol("@elapsed")) = $(getfield(CUDAdrv, Symbol("@elapsed")))
@eval $(Symbol("@profile")) = $(getfield(CUDAdrv, Symbol("@profile")))


@reexport using CUDAnative

## math intrinsics
for intr in [:cos, :cos_fast, :cospi, :sin, :sin_fast, :sinpi, :tan, :tan_fast, :acos,
             :asin, :atan, :atan2, :angle, :cosh, :sinh, :tanh, :acosh, :asinh, :atanh,
             :log, :log_fast, :log, :log_fast, :log10, :log10_fast, :log1p, :log2,
             :log2_fast, :logb, :ilogb, :exp, :exp_fast, :exp2, :exp10, :exp10_fast, :expm1,
             :ldexp, :exp, :exp_fast, :erf, :erfinv, :erfc, :erfcinv, :erfcx, :brev, :clz,
             :ffs, :byte_perm, :popc, :isfinite, :isinf, :isnan, :nearbyint, :nextafter,
             :signbit, :copysign, :abs, :sqrt, :rsqrt, :cbrt, :rcbrt, :pow, :pow_fast, :pow,
             :abs2, :ceil, :floor, :min, :max, :saturate, :mod, :rem, :div_fast, :lgamma,
             :tgamma, :j0, :j1, :jn, :y0, :y1, :yn, :normcdf, :normcdfinv, :hypot, :fma,
             :sad, :dim, :mul24, :mul64hi, :mulhi, :hadd, :rhadd, :scalbn]
    @eval const $intr = $(getfield(CUDAnative, intr))
end

## atomics
const atomic_xchg!  = CUDAnative.atomic_xchg!
const atomic_add!   = CUDAnative.atomic_add!
const atomic_and!   = CUDAnative.atomic_and!
const atomic_or!    = CUDAnative.atomic_or!
const atomic_xor!   = CUDAnative.atomic_xor!
const atomic_min!   = CUDAnative.atomic_min!
const atomic_max!   = CUDAnative.atomic_max!
const atomic_inc!   = CUDAnative.atomic_inc!
const atomic_dec!   = CUDAnative.atomic_dec!


@reexport using CuArrays

## array constructors
const zeros = CuArrays.zeros
const ones  = CuArrays.ones
const fill  = CuArrays.fill

## random numbers
const fill          = CuArrays.fill
const seed!         = CuArrays.seed!
const rand          = CuArrays.rand
const randn         = CuArrays.randn
const rand_logn     = CuArrays.rand_logn
const rand_poisson  = CuArrays.rand_poisson

@eval $(Symbol("@sync"))        = $(getfield(CuArrays, Symbol("@sync")))
@eval $(Symbol("@time"))        = $(getfield(CuArrays, Symbol("@time")))
@eval $(Symbol("@allocated"))   = $(getfield(CuArrays, Symbol("@allocated")))


end
