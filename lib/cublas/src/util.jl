const CublasFloat = Union{Float64,Float32,ComplexF64,ComplexF32}
const CublasReal = Union{Float64,Float32}
const CublasComplex = Union{ComplexF64,ComplexF32}

function Base.convert(::Type{cublasOperation_t}, trans::Char)
    if trans == 'N'
        return CUBLAS_OP_N
    elseif trans == 'T'
        return CUBLAS_OP_T
    elseif trans == 'C'
        return CUBLAS_OP_C
    else
        throw(ArgumentError("Unknown operation $trans"))
    end
end

function Base.convert(::Type{cublasFillMode_t}, uplo::Char)
    if uplo == 'U'
        return CUBLAS_FILL_MODE_UPPER
    elseif uplo == 'L'
        return CUBLAS_FILL_MODE_LOWER
    else
        throw(ArgumentError("Unknown fill mode $uplo"))
    end
end

function Base.convert(::Type{cublasDiagType_t}, diag::Char)
    if diag == 'U'
        return CUBLAS_DIAG_UNIT
    elseif diag == 'N'
        return CUBLAS_DIAG_NON_UNIT
    else
        throw(ArgumentError("Unknown diag mode $diag"))
    end
end

function Base.convert(::Type{cublasSideMode_t}, side::Char)
    if side == 'L'
        return CUBLAS_SIDE_LEFT
    elseif side == 'R'
        return CUBLAS_SIDE_RIGHT
    else
        throw(ArgumentError("Unknown side mode $side"))
    end
end
