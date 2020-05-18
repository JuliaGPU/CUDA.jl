# descriptor

mutable struct DropoutDesc
  ptr::Ptr{Nothing}
  states::CuVector{UInt8}
end

Base.unsafe_convert(::Type{Ptr{Nothing}}, dd::DropoutDesc) = dd.ptr

function DropoutDesc(ρ::Real; seed::Integer=0)
  d = [C_NULL]
  s = Csize_t[0]
  cudnnCreateDropoutDescriptor(d)
  cudnnDropoutGetStatesSize(handle(), s)
  states = CuArray{UInt8}(undef, s[]) # TODO: can we drop this when ρ=0?
  desc = DropoutDesc(d[], states)
  cudnnSetDropoutDescriptor(desc, handle(), ρ, states, length(states), seed)
  finalizer(desc) do x
    cudnnDestroyDropoutDescriptor(x)
  end
  return desc
end
