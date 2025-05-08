module SparseMatricesCSRExt

using CUDA
import CUDA.CUSPARSE:
    CuSparseMatrixCSR, CuSparseMatrixCSC, CuSparseMatrixCOO, CuSparseMatrixBSR,
    SparseMatrixCSC
using SparseMatricesCSR
import SparseMatricesCSR: SparseMatrixCSR
import Adapt

# CPU → GPU
CUSPARSE.CuSparseMatrixCSR{T}(Mat::SparseMatrixCSR) where {T} =
    CUSPARSE.CuSparseMatrixCSR{T}(
    CuVector{Cint}(Mat.rowptr), CuVector{Cint}(Mat.colval),
    CuVector{T}(Mat.nzval), size(Mat)
)
CUSPARSE.CuSparseMatrixCSC{T}(Mat::SparseMatrixCSR) where {T} = CuSparseMatrixCSC(CuSparseMatrixCSR{T}(Mat))
CUSPARSE.CuSparseMatrixCOO{T}(Mat::SparseMatrixCSR) where {T} = CuSparseMatrixCOO(CuSparseMatrixCSR{T}(Mat))
CUSPARSE.CuSparseMatrixBSR{T}(Mat::SparseMatrixCSR, blockdim) where {T} = CuSparseMatrixBSR(CuSparseMatrixCSR{T}(Mat), blockdim)

# GPU → CPU
SparseMatricesCSR.SparseMatrixCSR(A::CUSPARSE.CuSparseMatrixCSR) = SparseMatrixCSR{1}(size(A)..., Array(A.rowPtr), Array(A.colVal), Array(A.nzVal))
SparseMatricesCSR.SparseMatrixCSR(A::CUSPARSE.CuSparseMatrixCOO) = SparseMatrixCSR(CuSparseMatrixCSR(A))
SparseMatricesCSR.SparseMatrixCSR(A::CUSPARSE.CuSparseMatrixCSC) = SparseMatrixCSR(CuSparseMatrixCSR(A))
SparseMatricesCSR.SparseMatrixCSR(A::CUSPARSE.CuSparseMatrixBSR) = SparseMatrixCSR(CuSparseMatrixCSR(A))

# Adapt
Adapt.adapt_storage(::Type{CuArray}, xs::SparseMatrixCSR) = CUSPARSE.CuSparseMatrixCSR(xs)
Adapt.adapt_storage(::Type{CuArray{T}}, xs::SparseMatrixCSR) where {T} = CUSPARSE.CuSparseMatrixCSR{T}(xs)
Adapt.adapt_storage(::Type{Array}, mat::CUSPARSE.CuSparseMatrixCSR) = SparseMatrixCSR(mat)

end
