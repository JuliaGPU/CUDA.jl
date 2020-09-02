# broadcasting

using Base.Broadcast: BroadcastStyle, Broadcasted

struct CuArrayStyle{N} <: AbstractGPUArrayStyle{N} end
CuArrayStyle(::Val{N}) where N = CuArrayStyle{N}()
CuArrayStyle{M}(::Val{N}) where {N,M} = CuArrayStyle{N}()

BroadcastStyle(::Type{<:CuArray{T,N}}) where {T,N} = CuArrayStyle{N}()

Base.similar(bc::Broadcasted{CuArrayStyle{N}}, ::Type{T}) where {N,T} =
    similar(CuArray{T}, axes(bc))

Base.similar(bc::Broadcasted{CuArrayStyle{N}}, ::Type{T}, dims) where {N,T} =
    CuArray{T}(undef, dims)


## replace base functions with libdevice alternatives

cufunc(f) = f
cufunc(::Type{T}) where T = (x...) -> T(x...) # broadcasting type ctors isn't GPU compatible

Broadcast.broadcasted(::CuArrayStyle{N}, f, args...) where {N} =
  Broadcasted{CuArrayStyle{N}}(cufunc(f), args, nothing)

const device_intrinsics = :[
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

for f in device_intrinsics
  isdefined(Base, f) || continue
  @eval cufunc(::typeof(Base.$f)) = $f
end

# broadcast ^

culiteral_pow(::typeof(^), x::T, ::Val{0}) where {T<:Real} = one(x)
culiteral_pow(::typeof(^), x::T, ::Val{1}) where {T<:Real} = x
culiteral_pow(::typeof(^), x::T, ::Val{2}) where {T<:Real} = x * x
culiteral_pow(::typeof(^), x::T, ::Val{3}) where {T<:Real} = x * x * x
culiteral_pow(::typeof(^), x::T, ::Val{p}) where {T<:Real,p} = pow(x, Int32(p))

cufunc(::typeof(Base.literal_pow)) = culiteral_pow
cufunc(::typeof(Base.:(^))) = pow

using MacroTools

const _cufuncs = [copy(device_intrinsics); :^]
cufuncs() = (global _cufuncs; _cufuncs)

_cuint(x::Int) = Int32(x)
_cuint(x::Expr) = x.head == :call && x.args[1] == :Int32 && x.args[2] isa Int ? Int32(x.args[2]) : x
_cuint(x) = x

function _cupowliteral(x::Expr)
  if x.head == :call && x.args[1] == :(CUDA.cufunc(^)) && x.args[3] isa Int32
    num = x.args[3]
    if 0 <= num <= 3
      sym = gensym(:x)
      new_x = Expr(:block, :($sym = $(x.args[2])))

      if iszero(num)
        push!(new_x.args, :(one($sym)))
      else
        unroll = Expr(:call, :*)
        for x = one(num):num
          push!(unroll.args, sym)
        end
        push!(new_x.args, unroll)
      end

      x = new_x
    end
  end
  x
end
_cupowliteral(x) = x

function replace_device(ex)
  global _cufuncs
  MacroTools.postwalk(ex) do x
    x = x in _cufuncs ? :(CUDA.cufunc($x)) : x
    x = _cuint(x)
    x = _cupowliteral(x)
    x
  end
end

_cufunc(mod,ex) = begin
    global _cufuncs
    def = MacroTools.splitdef(ex)
    f = def[:name]
    if ~isdefined(mod,f)                # definition check
        Core.eval(mod,ex)
    elseif ~@eval $mod.$f isa Function  # name check
        error("$f already has a value, can't define cufunc_$f")
    end
    def[:name] = Symbol(:cufunc_, f)    # As the definition not in CUDA, a longer prefix to prevent something like msum(x) = ....
    def[:body] = replace_device(def[:body])
    push!(_cufuncs, f)
    Core.eval(mod,MacroTools.combinedef(def)) # define the `cuxxx` in the module called this macro, I'm not sure about it, should it be in CUDA?
    @eval CUDA.cufunc(::typeof($mod.$f)) = $mod.$(def[:name])
end

macro cufunc(ex)
  return :(_cufunc($__module__,$(QuoteNode(ex))))
end

