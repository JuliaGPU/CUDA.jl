# math functionality corresponding to SpecialFunctions.jl


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


## gamma function

@device_override SpecialFunctions.loggamma(x::Float64) = ccall("extern __nv_lgamma", llvmcall, Cdouble, (Cdouble,), x)
@device_override SpecialFunctions.loggamma(x::Float32) = ccall("extern __nv_lgammaf", llvmcall, Cfloat, (Cfloat,), x)

@device_override SpecialFunctions.gamma(x::Float64) = ccall("extern __nv_tgamma", llvmcall, Cdouble, (Cdouble,), x)
@device_override SpecialFunctions.gamma(x::Float32) = ccall("extern __nv_tgammaf", llvmcall, Cfloat, (Cfloat,), x)

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
