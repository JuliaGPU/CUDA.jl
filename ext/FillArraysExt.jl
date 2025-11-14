module FillArraysExt

using CUDA
using CUDA.CUSPARSE
using LinearAlgebra
using SparseArrays

isdefined(Base, :get_extension) ? (using FillArrays) : (using ..FillArrays)

# kron between CuSparseMatrixCOO and Diagonal{T, AbstractFill}
# This is optimized for FillArrays since the diagonal is a constant value
function LinearAlgebra.kron(A::CUSPARSE.CuSparseMatrixCOO{T1, Ti}, B::Diagonal{T2, <:FillArrays.AbstractFill{T2}}) where {Ti, T1, T2}
    T = promote_type(T1, T2)
    mA, nA = size(A)
    mB, nB = size(B)
    out_shape = (mA * mB, nA * nB)
    Annz = Int64(A.nnz)
    Bnnz = nB

    if Annz == 0 || Bnnz == 0
        return CUSPARSE.CuSparseMatrixCOO(CUDA.CuVector{Ti}(undef, 0), CUDA.CuVector{Ti}(undef, 0), CUDA.CuVector{T}(undef, 0), out_shape)
    end

    # Get the fill value from the diagonal
    fill_value = FillArrays.getindex_value(B.diag)

    row = (A.rowInd .- 1) .* mB
    row = repeat(row, inner = Bnnz)
    col = (A.colInd .- 1) .* nB
    col = repeat(col, inner = Bnnz)
    data = repeat(convert(CUDA.CuVector{T}, A.nzVal), inner = Bnnz)

    row .+= CUDA.CuVector(repeat(0:nB-1, outer = Annz)) .+ 1
    col .+= CUDA.CuVector(repeat(0:nB-1, outer = Annz)) .+ 1

    # Multiply by the fill value (already promoted type T)
    data .*= fill_value

    CUSPARSE.sparse(row, col, data, out_shape..., fmt = :coo)
end

# kron between Diagonal{T, AbstractFill} and CuSparseMatrixCOO
function LinearAlgebra.kron(A::Diagonal{T1, <:FillArrays.AbstractFill{T1}}, B::CUSPARSE.CuSparseMatrixCOO{T2, Ti}) where {Ti, T1, T2}
    T = promote_type(T1, T2)
    mA, nA = size(A)
    mB, nB = size(B)
    out_shape = (mA * mB, nA * nB)
    Annz = nA
    Bnnz = Int64(B.nnz)

    if Annz == 0 || Bnnz == 0
        return CUSPARSE.CuSparseMatrixCOO(CUDA.CuVector{Ti}(undef, 0), CUDA.CuVector{Ti}(undef, 0), CUDA.CuVector{T}(undef, 0), out_shape)
    end

    # Get the fill value from the diagonal
    fill_value = FillArrays.getindex_value(A.diag)

    row = (0:nA-1) .* mB
    row = CUDA.CuVector(repeat(row, inner = Bnnz))
    col = (0:nA-1) .* nB
    col = CUDA.CuVector(repeat(col, inner = Bnnz))
    data = CUDA.fill(T(fill_value), nA * Bnnz)

    row .+= repeat(B.rowInd .- 1, outer = Annz) .+ 1
    col .+= repeat(B.colInd .- 1, outer = Annz) .+ 1

    data .*= repeat(B.nzVal, outer = Annz)

    CUSPARSE.sparse(row, col, data, out_shape..., fmt = :coo)
end

end  # extension module
