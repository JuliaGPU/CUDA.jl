@enum libraryPropertyType::Cint begin
    MAJOR_VERSION = 0
    MINOR_VERSION
    PATCH_LEVEL
end

@enum cudaDataType::Cint begin
    R_16F  =  2 # real as a half
    C_16F  =  6 # complex as a pair of half numbers
    R_16BF = 14 # real as a nv_bfloat16
    C_16BF = 15 # complex as a pair of nv_bfloat16 numbers
    R_32F  =  0 # real as a float
    C_32F  =  4 # complex as a pair of float numbers
    R_64F  =  1 # real as a double
    C_64F  =  5 # complex as a pair of double numbers
    R_4I   = 16 # real as a signed 4-bit int
    C_4I   = 17 # complex as a pair of signed 4-bit int numbers
    R_4U   = 18 # real as a unsigned 4-bit int
    C_4U   = 19 # complex as a pair of unsigned 4-bit int numbers
    R_8I   =  3 # real as a signed 8-bit int
    C_8I   =  7 # complex as a pair of signed 8-bit int numbers
    R_8U   =  8 # real as a unsigned 8-bit int
    C_8U   =  9 # complex as a pair of unsigned 8-bit int numbers
    R_16I  = 20 # real as a signed 16-bit int
    C_16I  = 21 # complex as a pair of signed 16-bit int numbers
    R_16U  = 22 # real as a unsigned 16-bit int
    C_16U  = 23 # complex as a pair of unsigned 16-bit int numbers
    R_32I  = 10 # real as a signed 32-bit int
    C_32I  = 11 # complex as a pair of signed 32-bit int numbers
    R_32U  = 12 # real as a unsigned 32-bit int
    C_32U  = 13 # complex as a pair of unsigned 32-bit int numbers
    R_64I  = 24 # real as a signed 64-bit int
    C_64I  = 25 # complex as a pair of signed 64-bit int numbers
    R_64U  = 26 # real as a unsigned 64-bit int
    C_64U  = 27 # complex as a pair of unsigned 64-bit int numbers
end

function Base.convert(::Type{cudaDataType}, T::DataType)
    if T === Float16
        return R_16F
    elseif T === Complex{Float16}
        return C_16F
    elseif T === BFloat16
        return R_16BF
    elseif T === Complex{BFloat16}
        return C_16BF
    elseif T === Float32
        return R_32F
    elseif T === Complex{Float32}
        return C_32F
    elseif T === Float64
        return R_64F
    elseif T === Complex{Float64}
        return C_64F
    # elseif T === Int4
    #     return R_4I
    # elseif T === Complex{Int4}
    #     return C_4I
    # elseif T === UInt4
    #     return R_4U
    # elseif T === Complex{Int4}
    #     return C_4U
    elseif T === Int8
        return R_8I
    elseif T === Complex{Int8}
        return C_8I
    elseif T === UInt8
        return R_8U
    elseif T === Complex{UInt8}
        return C_8U
    elseif T === Int16
        return R_16I
    elseif T === Complex{Int16}
        return C_16I
    elseif T === UInt16
        return R_16U
    elseif T === Complex{UInt16}
        return C_16U
    elseif T === Int32
        return R_32I
    elseif T === Complex{Int32}
        return C_32I
    elseif T === UInt32
        return R_32U
    elseif T === Complex{UInt32}
        return C_32U
    elseif T === Int64
        return R_64I
    elseif T === Complex{Int64}
        return C_64I
    elseif T === UInt64
        return R_64U
    elseif T === Complex{UInt64}
        return C_64U
    else
        throw(ArgumentError("cudaDataType equivalent for input type $T does not exist!"))
    end
end

function Base.convert(::Type{Type}, T::cudaDataType)
    if T == R_16F
        return Float16
    elseif T == C_16F
        return Complex{Float16}
    elseif T == R_16BF
        return BFloat16
    elseif T == C_16BF
        return Complex{BFloat16}
    elseif T == R_32F
        return Float32
    elseif T == C_32F
        return Complex{Float32}
    elseif T == R_64F
        return Float64
    elseif T == C_64F
        return Complex{Float64}
    # elseif T == R_4I
    #     return Int4
    # elseif T == C_4I
    #     return Complex{Int4}
    # elseif T == R_4U
    #     return UInt4
    # elseif T == C_4U
    #     return Complex{Int4}
    elseif T == R_8I
        return Int8
    elseif T == C_8I
        return Complex{Int8}
    elseif T == R_8U
        return UInt8
    elseif T == C_8U
        return Complex{UInt8}
    elseif T == R_16I
        return Int16
    elseif T == C_16I
        return Complex{Int16}
    elseif T == R_16U
        return UInt16
    elseif T == C_16U
        return Complex{UInt16}
    elseif T == R_32I
        return Int32
    elseif T == C_32I
        return Complex{Int32}
    elseif T == R_32U
        return UInt32
    elseif T == C_32U
        return Complex{UInt32}
    elseif T == R_64I
        return Int64
    elseif T == C_64I
        return Complex{Int64}
    elseif T == R_64U
        return UInt64
    elseif T == C_64U
        return Complex{UInt64}
    else
        throw(ArgumentError("Julia type equivalent for input type $T does not exist!"))
    end
end
