@enum libraryPropertyType::Cint begin
    MAJOR_VERSION = 0
    MINOR_VERSION
    PATCH_LEVEL
end

@enum cudaDataType::Cint begin
    R_16F = 2  # `CUDA_R_16F`, real as a half
    C_16F = 6  # `CUDA_C_16F`, complex as a pair of half numbers
    R_32F = 0  # `CUDA_R_32F`, real as a float
    C_32F = 4  # `CUDA_C_32F`, complex as a pair of float numbers
    R_64F = 1  # `CUDA_R_64F`, real as a double
    C_64F = 5  # `CUDA_C_64F`, complex as a pair of double numbers
    R_8I  = 3  # `CUDA_R_8I`,  real as a signed char
    C_8I  = 7  # `CUDA_C_8I`,  complex as a pair of signed char numbers
    R_8U       # `CUDA_R_8U`,  real as a unsigned char
    C_8U       # `CUDA_C_8U`,  complex as a pair of unsigned char numbers
    R_32I      # `CUDA_R_32I`, real as a signed int
    C_32I      # `CUDA_C_32I`, complex as a pair of signed int numbers
    R_32U      # `CUDA_R_32U`, real as a unsigned int
    C_32U      # `CUDA_C_32U`, complex as a pair of unsigned int numbers
end

function cudaDataType(T::DataType)
    if T == Float32
        return R_32F
    elseif T == ComplexF32
        return C_32F
    elseif T == Float16
        return R_16F
    elseif T == ComplexF16
        return C_16F
    elseif T == Float64
        return R_64F
    elseif T == ComplexF64
        return C_64F
    elseif T == Int8
        return R_8I
    elseif T == Complex{Int8}
        return C_8I
    elseif T == Int32
        return R_32I
    elseif T == Complex{Int32}
        return C_32I
    elseif T == UInt8
        return R_8U
    elseif T == Complex{UInt8}
        return C_8U
    elseif T == UInt32
        return R_32U
    elseif T == Complex{UInt32}
        return C_32U
    else
        throw(ArgumentError("cudaDataType equivalent for input type $T does not exist!"))
    end
end
