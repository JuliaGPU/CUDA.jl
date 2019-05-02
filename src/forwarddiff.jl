# ForwardDiff integration

for f in libdevice
  if haskey(ForwardDiff.DiffRules.DEFINED_DIFFRULES, (:Base,f,1))
    f == :tanh && continue
    diffrule = ForwardDiff.DiffRules.DEFINED_DIFFRULES[(:Base,f,1)]
    ForwardDiff.DiffRules.DEFINED_DIFFRULES[(:CUDAnative,f,1)] =
      (args...) -> replace_device(diffrule(args...))
    eval(ForwardDiff.unary_dual_definition(:CUDAnative, f))
  end
end

ForwardDiff.DiffRules.DEFINED_DIFFRULES[(:CUDAnative, :tanh, 1)] = x ->
  replace_device(:(1-tanh(x)^2))
eval(ForwardDiff.unary_dual_definition(:CUDAnative, :tanh))

ForwardDiff.DiffRules.DEFINED_DIFFRULES[(:CUDAnative, :pow, 2)] = (x, y) ->
  replace_device.(ForwardDiff.DiffRules.DEFINED_DIFFRULES[(:Base, :^, 2)](x, y))

@eval begin
  ForwardDiff.@define_binary_dual_op(
      CUDAnative.pow,
      begin
        vx = ForwardDiff.value(x)
        vy = ForwardDiff.value(y)
        expv = (CUDAnative.pow)(vx, vy)

        powval = vy * CUDAnative.pow(vx , oftype(vx, vy - 1))

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
          new_partials = ForwardDiff.partials(x) * y * (CUDAnative.pow)(v, y - 1)
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
