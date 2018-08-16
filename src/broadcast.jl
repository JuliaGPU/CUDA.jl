import Base.Broadcast: Broadcasted, Extruded, BroadcastStyle, ArrayStyle, preprocess, preprocess_args

# GPUArrays.jl defines broadcast for us and we only need to ensure that Broadcast/Extruded gets converted
# to variants that are valid on the GPU, as an example we need to convert CuArray to CuDeviceArray
cudaconvert(bc::Broadcasted{Style}) where Style = Broadcasted{Style}(bc.f, map(cudaconvert, bc.args), bc.axes)
cudaconvert(ex::Extruded) = Extruded(cudaconvert(ex.x), ex.keeps, ex.defaults)
cudaconvert(x::LinearAlgebra.Transpose{<:Any,<:CuArray}) = LinearAlgebra.Transpose(cudaconvert(x.vec))

# Ref{CuArray} is invalid for GPU codegen
# see https://github.com/JuliaGPU/CUDAnative.jl/issues/223
# so we do a read only broadcast ref
struct CuRefValue{T} <: Ref{T}
  x::T
end
Base.getindex(r::CuRefValue) = r.x
cudaconvert(r::Ref) = CuRefValue(cudaconvert(r[]))

# Until we can use Cassette to do this translation for use we **try** to do some manually fixing
import NNlib: @fix, _cufunc

_cufunc(f,x::CuArray,xs...) = cufunc(f)
cufunc(x) = x

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
  @eval begin
    cufunc(::typeof(Base.$f)) = CUDAnative.$f
    @inline preprocess(dest::CuArray, bc::Broadcasted{Nothing,<:Any,typeof(Base.$f)}) = Broadcasted{Nothing}(CUDAnative.$f, preprocess_args(dest, bc.args), bc.axes)
  end
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
    @inline function Base.Broadcast.preprocess(dest::CuArrays.CuArray, bc::Base.Broadcast.Broadcasted{Nothing,<:Any,typeof($(esc(f)))})
      Base.Broadcast.Broadcasted{Nothing}($(esc(def[:name])), Base.Broadcast.preprocess_args(dest, bc.args), bc.axes)
    end
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
