# compatibility with StaticArrays

module StaticArraysExt

using ..CUDACore
using ..CUDACore: @device_override, @gputhrow

import StaticArrays

# same quirk as for some Base methods in src/device/quirks.jl
@device_override @noinline StaticArrays.dimension_mismatch_fail(::Type{SA}, a::AbstractArray) where {SA<:StaticArrays.StaticArray} =
    @gputhrow("DimensionMismatch", "DimensionMismatch while trying to convert to StaticArray: Expected and actual length of input array differ.")

end  # extension module
