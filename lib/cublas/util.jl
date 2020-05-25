# convert matrix to band storage
function band(A::AbstractMatrix,kl,ku)
    m, n = size(A)
    AB = zeros(eltype(A),kl+ku+1,n)
    for j = 1:n
        for i = max(1,j-ku):min(m,j+kl)
            AB[ku+1-j+i,j] = A[i,j]
        end
    end
    return AB
end

# convert band storage to general matrix
function unband(AB::AbstractMatrix,m,kl,ku)
    bm, n = size(AB)
    A = zeros(eltype(AB),m,n)
    for j = 1:n
        for i = max(1,j-ku):min(m,j+kl)
            A[i,j] = AB[ku+1-j+i,j]
        end
    end
    return A
end

# zero out elements not on matrix bands
function bandex(A::AbstractMatrix,kl,ku)
    m, n = size(A)
    AB = band(A,kl,ku)
    B = unband(AB,m,kl,ku)
    return B
end

const CublasFloat = Union{Float64,Float32,ComplexF64,ComplexF32}
const CublasReal = Union{Float64,Float32}
const CublasComplex = Union{ComplexF64,ComplexF32}

# convert Char {N,T,C} to cublasOperation_t
function cublasop(trans::Char)
    if trans == 'N'
        return CUBLAS_OP_N
    end
    if trans == 'T'
        return CUBLAS_OP_T
    end
    if trans == 'C'
        return CUBLAS_OP_C
    end
    throw(ArgumentError("unknown cublas operation $trans"))
end

# convert Char {U,L} to cublasFillMode_t
function cublasfill(uplo::Char)
    if uplo == 'U'
        return CUBLAS_FILL_MODE_UPPER
    end
    if uplo == 'L'
        return CUBLAS_FILL_MODE_LOWER
    end
    throw(ArgumentError("unknown cublas fill mode $uplo"))
end

# convert Char {U,N} to cublasDiagType_t
function cublasdiag(diag::Char)
    if diag == 'U'
        return CUBLAS_DIAG_UNIT
    end
    if diag == 'N'
        return CUBLAS_DIAG_NON_UNIT
    end
    throw(ArgumentError("unknown cublas diag mode $diag"))
end

# convert Char {L,R}
function cublasside(side::Char)
    if side == 'L'
        return CUBLAS_SIDE_LEFT
    end
    if side == 'R'
        return CUBLAS_SIDE_RIGHT
    end
    throw(ArgumentError("unknown cublas side mode $side"))
end
