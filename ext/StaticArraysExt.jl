# compatibility with StaticArrays

module StaticArraysExt

using ..CUDA
using ..CUDA: @device_override, @print_and_throw

import StaticArrays

# same quirk as for some Base methods in src/device/quirks.jl
@device_override @noinline StaticArrays.dimension_mismatch_fail(::Type{SA}, a::AbstractArray) where {SA<:StaticArrays.StaticArray} =
    @print_and_throw("DimensionMismatch while trying to convert to StaticArray: Expected and actual length of input array differ.")

end  # extension module