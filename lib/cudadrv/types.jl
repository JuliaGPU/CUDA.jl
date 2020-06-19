export CuDim3, CuDim

"""
    CuDim3(x)

    CuDim3((x,))
    CuDim3((x, y))
    CuDim3((x, y, x))

A type used to specify dimensions, consisting of 3 integers for respectively the `x`, `y`
and `z` dimension. Unspecified dimensions default to `1`.

Often accepted as argument through the `CuDim` type alias, eg. in the case of
[`cudacall`](@ref) or [`launch`](@ref), allowing to pass dimensions as a plain integer or a
tuple without having to construct an explicit `CuDim3` object.
"""
struct CuDim3
    x::Cuint
    y::Cuint
    z::Cuint
end

CuDim3(dims::Integer)             = CuDim3(dims,    Cuint(1), Cuint(1))
CuDim3(dims::NTuple{1,<:Integer}) = CuDim3(dims[1], Cuint(1), Cuint(1))
CuDim3(dims::NTuple{2,<:Integer}) = CuDim3(dims[1], dims[2],  Cuint(1))
CuDim3(dims::NTuple{3,<:Integer}) = CuDim3(dims[1], dims[2],  dims[3])

# Type alias for conveniently specifying the dimensions
# (e.g. `(len, 2)` instead of `CuDim3((len, 2))`)
const CuDim = Union{Integer,
                    Tuple{Integer},
                    Tuple{Integer, Integer},
                    Tuple{Integer, Integer, Integer}}
