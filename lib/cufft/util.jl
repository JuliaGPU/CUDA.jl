const cufftReals = Union{cufftDoubleReal,cufftReal,Float16}
const cufftComplexes = Union{cufftDoubleComplex,cufftComplex,Complex{Float16}}
const cufftNumber = Union{cufftReals,cufftComplexes}

cufftfloat(x) = _cufftfloat(float(x))
_cufftfloat(::Type{T}) where {T<:cufftNumber} = T
_cufftfloat(::Type{T}) where {T} = error("type $T not supported")
_cufftfloat(x::T) where {T} = _cufftfloat(T)(x)

realfloat(x::DenseCuArray{<:cufftReals}) = x
realfloat(x::DenseCuArray{T}) where {T<:Real} = copy1(cufftfloat(T), x)
realfloat(x::DenseCuArray{T}) where {T} = error("type $T not supported")

complexfloat(x::DenseCuArray{<:cufftComplexes}) = x
complexfloat(x::DenseCuArray{T}) where {T<:Complex} = copy1(cufftfloat(T), x)
complexfloat(x::DenseCuArray{T}) where {T<:Real} = copy1(cufftfloat(complex(T)), x)
complexfloat(x::DenseCuArray{T}) where {T} = error("type $T not supported")

function copy1(::Type{T}, x) where T
    y = CuArray{T}(undef, map(length, axes(x)))
    #copy!(y, x)
    y .= broadcast(xi->convert(T,xi),x)
end
