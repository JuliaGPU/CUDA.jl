const cufftReals = Union{cufftDoubleReal,cufftReal,Float16}
const cufftComplexes = Union{cufftDoubleComplex,cufftComplex,Complex{Float16}}
const cufftNumber = Union{cufftReals,cufftComplexes}

cufftfloat(x) = _cufftfloat(float(x))
_cufftfloat(::Type{T}) where {T<:cufftReals} = T
_cufftfloat(::Type{Float16}) = Float32
_cufftfloat(::Type{Complex{T}}) where {T} = Complex{_cufftfloat(T)}
_cufftfloat(::Type{T}) where {T} = error("type $T not supported")
_cufftfloat(x::T) where {T} = _cufftfloat(T)(x)

complexfloat(x::DenseCuArray{Complex{<:cufftReals}}) = x
realfloat(x::DenseCuArray{<:cufftReals}) = x

complexfloat(x::DenseCuArray{T}) where {T<:Complex} = copy1(typeof(cufftfloat(zero(T))), x)
complexfloat(x::DenseCuArray{T}) where {T<:Real} = copy1(typeof(complex(cufftfloat(zero(T)))), x)

realfloat(x::DenseCuArray{T}) where {T<:Real} = copy1(typeof(cufftfloat(zero(T))), x)

function _cufftType(input_type::Type{<:cufftNumber}, output_type::Type{<:cufftNumber})
    input_type == cufftReal && output_type == cufftComplex && return CUFFT_R2C
    input_type == cufftComplex && output_type == cufftReal && return CUFFT_C2R
    input_type == cufftComplex && output_type == cufftComplex && return CUFFT_C2C
    input_type == cufftDoubleReal && output_type == cufftDoubleComplex && return CUFFT_D2Z
    input_type == cufftDoubleComplex && output_type == cufftDoubleReal && return CUFFT_Z2D
    input_type == cufftDoubleComplex && output_type == cufftDoubleComplex && return CUFFT_Z2Z
    throw(ArgumentError("There is no cufftType for input_type=$input_type and output_type=$output_type"))
end

function copy1(::Type{T}, x) where T
    y = CuArray{T}(undef, map(length, axes(x)))
    #copy!(y, x)
    y .= broadcast(xi->convert(T,xi),x)
end
