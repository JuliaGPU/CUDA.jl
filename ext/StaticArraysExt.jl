# compatibility with StaticArrays

module StaticArraysExt

using CUDA: @device_override, @print_and_throw

isdefined(Base, :get_extension) ? (import StaticArrays) : (import ..StaticArrays)

# same quirk as for some Base methods in src/device/quirks.jl
@device_override @noinline StaticArrays.dimension_mismatch_fail(::Type{SA}, a::AbstractArray) where {SA<:StaticArray} =
    @print_and_throw(" DimensionMismatch: Expected and actual length of input array differ.")

end  # extension module