# using Base.Cartesian
using LinearAlgebra: Transpose
import Base.Broadcast: Broadcasted, Extruded, BroadcastStyle, ArrayStyle

cudaconvert(bc::Broadcasted{Style}) where Style = Broadcasted{Style}(bc.axes, map(cudaconvert, bc.args), bc.axes)
cudaconvert(ex::Extruded) = Extruded(cudaconvert(ex.x), e.keeps, ex.defaults)
cudaconvert(x::Transpose{<:Any,<:CuArray}) = Transpose(cudaconvert(x.vec))

BroadcastStyle(::Type{<:CuArray}) = ArrayStyle{CuArray}()
BroadcastStyle(::Type{<:Transpose{<:Any,<:CuArray}}) = ArrayStyle{CuArray}()
Base.similar(bc::Broadcasted{ArrayStyle{CuArray}}, ::Type{ET}) where ET = similar(CuArray{ET}, axes(bc))

# Broadcast function fixes

import NNlib: @fix, _cufunc

_cufunc(f,x::CuArray,xs...) = cufunc(f)

cufunc(x) = x

libdevice = :[
  cos, cospi, sin, sinpi, tan, acos, asin, atan, atan2,
  cosh, sinh, tanh, acosh, asinh, atanh,
  log, log10, log1p, log2, logb, ilogb,
  exp, exp2, exp10, expm1, ldexp,
  erf, erfinv, erfc, erfcinv, erfcx,
  brev, clz, ffs, byte_perm, popc,
  isfinite, isinf, isnan, nearbyint,
  nextafter, signbit, copysign, abs,
  sqrt, rsqrt, cbrt, rcbrt, pow,
  ceil, floor, saturate,
  lgamma, tgamma,
  j0, j1, jn, y0, y1, yn,
  normcdf, normcdfinv, hypot,
  fma, sad, dim, mul24, mul64hi, hadd, rhadd, scalbn].args

for f in libdevice
  # TODO use Broadcast.broadcasted(::ArrayStyle{<:CuArray}, ::typeof(f), args...)
  isdefined(Base, f) || continue
  @eval cufunc(::typeof(Base.$f)) = CUDAnative.$f
end

# ForwardDiff Integration

using MacroTools

function replace_device(ex)
  MacroTools.postwalk(ex) do x
    x in libdevice ? :(CUDAnative.$x) : x
  end
end

macro cufunc(ex)
  def = MacroTools.splitdef(ex)
  f = def[:name]
  def[:name] = Symbol(:cu, f)
  def[:body] = replace_device(def[:body])
  quote
    $(esc(MacroTools.combinedef(def)))
    CuArrays.cufunc(::typeof($(esc(f)))) = $(esc(def[:name]))
  end
end

using ForwardDiff: Dual, value, partials, unary_dual_definition
using DiffRules

for f in libdevice
  if haskey(DiffRules.DEFINED_DIFFRULES, (:Base,f,1))
    f == :tanh && continue
    diffrule = DiffRules.DEFINED_DIFFRULES[(:Base,f,1)]
    DiffRules.DEFINED_DIFFRULES[(:CUDAnative,f,1)] =
      (args...) -> replace_device(diffrule(args...))
    eval(unary_dual_definition(:CUDAnative, f))
  end
end

DiffRules.DEFINED_DIFFRULES[(:CUDAnative, :tanh, 1)] = x ->
  replace_device(:(1-tanh(x)^2))
eval(unary_dual_definition(:CUDAnative, :tanh))
