import Base.Broadcast: Broadcasted, Extruded, BroadcastStyle, ArrayStyle

BroadcastStyle(::Type{<:CuArray}) = ArrayStyle{CuArray}()

function Base.similar(bc::Broadcasted{ArrayStyle{CuArray}}, ::Type{T}) where T
    similar(CuArray, T, axes(bc))
end

# GPUArrays.jl defines broadcast for us and we only need to ensure that Broadcast/Extruded gets converted
# to variants that are valid on the GPU, as an example we need to convert CuArray to CuDeviceArray
cudaconvert(bc::Broadcasted{Style}) where Style = Broadcasted{Style}(bc.f, map(cudaconvert, bc.args), bc.axes)
cudaconvert(ex::Extruded) = Extruded(cudaconvert(ex.x), ex.keeps, ex.defaults)
cudaconvert(x::LinearAlgebra.Transpose{<:Any,<:CuArray}) = LinearAlgebra.Transpose(cudaconvert(x.parent))

# Ref{CuArray} is invalid for GPU codegen
# see https://github.com/JuliaGPU/CUDAnative.jl/issues/223
# so we do a read only broadcast ref
struct CuRefValue{T} <: Ref{T}
  x::T
end
Base.getindex(r::CuRefValue) = r.x
cudaconvert(r::Base.RefValue) = CuRefValue(cudaconvert(r[]))

# Until we can use Cassette to do this translation for use we **try** to do some manually fixing

cufunc(f) = f

Broadcast.broadcasted(::ArrayStyle{CuArray}, f, args...) =
  Broadcasted{ArrayStyle{CuArray}}(cufunc(f), args, nothing)

libdevice = :[
  cos, cospi, sin, sinpi, tan, acos, asin, atan,
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
  isdefined(Base, f) || continue
  @eval cufunc(::typeof(Base.$f)) = CUDAnative.$f
end

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

# ForwardDiff Integration
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
