module SparseMatricesCSRExt

using CUDA
import CUDA: CUSPARSE, CuVector
using SparseMatricesCSR
import SparseMatricesCSR: SparseMatrixCSR
import Adapt

CUSPARSE.CuSparseMatrixCSR{T}(Mat::SparseMatrixCSR) where {T} =
    CUSPARSE.CuSparseMatrixCSR{T}(
    CuVector{Cint}(Mat.rowptr), CuVector{Cint}(Mat.colval),
    CuVector{T}(Mat.nzval), size(Mat)
)

CUSPARSE.CuSparseMatrixCSC{T}(Mat::SparseMatrixCSR) where {T} =
    CUSPARSE.CuSparseMatrixCSC{T}(CUSPARSE.CuSparseMatrixCSR(Mat))

SparseMatricesCSR.SparseMatrixCSR(A::CUSPARSE.CuSparseMatrixCSR) =
    SparseMatrixCSR(CUSPARSE.SparseMatrixCSC(A)) # no direct conversion (gpu_CSR -> cpu_CSC -> cpu_CSR)

Adapt.adapt_storage(::Type{CUDA.CuArray}, xs::SparseMatrixCSR) =
    CUSPARSE.CuSparseMatrixCSR(xs)

Adapt.adapt_storage(::Type{CuArray{T}}, xs::SparseMatrixCSR) where {T} = CUSPARSE.CuSparseMatrixCSR{T}(xs)

Adapt.adapt_storage(::Type{Array}, mat::CUSPARSE.CuSparseMatrixCSR) =
    SparseMatrixCSR(mat)

end
