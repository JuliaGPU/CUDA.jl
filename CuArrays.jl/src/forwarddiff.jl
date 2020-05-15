# ForwardDiff integration

byhand = [:exp2, :log2, :exp10, :log10, :abs]

for f in libdevice
  if haskey(ForwardDiff.DiffRules.DEFINED_DIFFRULES, (:Base,f,1))
    f ∈ byhand && continue
    diffrule = ForwardDiff.DiffRules.DEFINED_DIFFRULES[(:Base,f,1)]
    ForwardDiff.DiffRules.DEFINED_DIFFRULES[(:CUDAnative,f,1)] =
      (args...) -> replace_device(diffrule(args...))
    eval(ForwardDiff.unary_dual_definition(:CUDAnative, f))
  end
end

# byhand: exp2
ForwardDiff.DiffRules.DEFINED_DIFFRULES[(:CUDAnative, :exp2, 1)] = x ->
  :((CuArrays.cufunc(exp2))(x) * (CuArrays.cufunc(log))(oftype(x, 2)))
eval(ForwardDiff.unary_dual_definition(:CUDAnative, :exp2))

# byhand: log2
ForwardDiff.DiffRules.DEFINED_DIFFRULES[(:CUDAnative, :log2, 1)] = x ->
   :(inv(x) / (CuArrays.cufunc(log))(oftype(x, 2)))
eval(ForwardDiff.unary_dual_definition(:CUDAnative, :log2))

# byhand: exp10
ForwardDiff.DiffRules.DEFINED_DIFFRULES[(:CUDAnative, :exp10, 1)] = x ->
  :((CuArrays.cufunc(exp10))(x) * (CuArrays.cufunc(log))(oftype(x, 10)))
eval(ForwardDiff.unary_dual_definition(:CUDAnative, :exp10))

# byhand: log10
ForwardDiff.DiffRules.DEFINED_DIFFRULES[(:CUDAnative, :log10, 1)] = x ->
   :(inv(x) / (CuArrays.cufunc(log))(oftype(x, 10)))
eval(ForwardDiff.unary_dual_definition(:CUDAnative, :log10))

# byhand: abs
ForwardDiff.DiffRules.DEFINED_DIFFRULES[(:CUDAnative, :abs, 1)] = x ->
   :(signbit(x) ? -one(x) : one(x))
eval(ForwardDiff.unary_dual_definition(:CUDAnative, :abs))


ForwardDiff.DiffRules.DEFINED_DIFFRULES[(:CUDAnative, :pow, 2)] = (x, y) ->
  replace_device.(ForwardDiff.DiffRules.DEFINED_DIFFRULES[(:Base, :^, 2)](x, y))

@eval begin
  ForwardDiff.@define_binary_dual_op(
      CUDAnative.pow,
      begin
        vx = ForwardDiff.value(x)
        vy = ForwardDiff.value(y)
        expv = (CUDAnative.pow)(vx, vy)

        powval = vy * CUDAnative.pow(vx , vy - Int32(1))

        py = ForwardDiff.partials(y)
        px = ForwardDiff.partials(x)

        cond = all(py.values) do x
          x == zero(x)
        end

        if cond
          logval = one(expv)
        else
          logval = expv * CUDAnative.log(vx)
        end

        new_partials = powval * px + logval * py
        return ForwardDiff.Dual{Txy}(expv, new_partials)
      end,
      begin
        v = ForwardDiff.value(x)
        expv = (CUDAnative.pow)(v, y)
        if y == zero(y)
          new_partials = zero(ForwardDiff.partials(x))
        else
          new_partials = ForwardDiff.partials(x) * y * (CUDAnative.pow)(v, y - Int32(1))
        end
        return ForwardDiff.Dual{Tx}(expv, new_partials)
      end,
      begin
        v = ForwardDiff.value(y)
        expv = (CUDAnative.pow)(x, v)
        deriv = expv*CUDAnative.log(x)
        return ForwardDiff.Dual{Ty}(expv, deriv * ForwardDiff.partials(y))
      end
    )
end
