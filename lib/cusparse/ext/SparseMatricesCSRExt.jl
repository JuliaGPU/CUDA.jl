module SparseMatricesCSRExt

using CUDA
using cuSPARSE
import cuSPARSE:
    CuSparseMatrixCSR, CuSparseMatrixCSC, CuSparseMatrixCOO, CuSparseMatrixBSR
using SparseMatricesCSR
import SparseMatricesCSR: SparseMatrixCSR
import Adapt

# CPU → GPU
cuSPARSE.CuSparseMatrixCSR{T}(Mat::SparseMatrixCSR) where {T} =
    cuSPARSE.CuSparseMatrixCSR{T}(
    CuVector{Cint}(Mat.rowptr), CuVector{Cint}(Mat.colval),
    CuVector{T}(Mat.nzval), size(Mat)
)
cuSPARSE.CuSparseMatrixCSC{T}(Mat::SparseMatrixCSR) where {T} = CuSparseMatrixCSC(CuSparseMatrixCSR{T}(Mat))
cuSPARSE.CuSparseMatrixCOO{T}(Mat::SparseMatrixCSR) where {T} = CuSparseMatrixCOO(CuSparseMatrixCSR{T}(Mat))
cuSPARSE.CuSparseMatrixBSR{T}(Mat::SparseMatrixCSR, blockdim) where {T} = CuSparseMatrixBSR(CuSparseMatrixCSR{T}(Mat), blockdim)

# GPU → CPU
SparseMatricesCSR.SparseMatrixCSR(A::cuSPARSE.CuSparseMatrixCSR) = SparseMatrixCSR{1}(size(A)..., Array(A.rowPtr), Array(A.colVal), Array(A.nzVal))
SparseMatricesCSR.SparseMatrixCSR(A::cuSPARSE.CuSparseMatrixCOO) = SparseMatrixCSR(CuSparseMatrixCSR(A))
SparseMatricesCSR.SparseMatrixCSR(A::cuSPARSE.CuSparseMatrixCSC) = SparseMatrixCSR(CuSparseMatrixCSR(A))
SparseMatricesCSR.SparseMatrixCSR(A::cuSPARSE.CuSparseMatrixBSR) = SparseMatrixCSR(CuSparseMatrixCSR(A))

# Adapt
Adapt.adapt_storage(::Type{CuArray}, xs::SparseMatrixCSR) = cuSPARSE.CuSparseMatrixCSR(xs)
Adapt.adapt_storage(::Type{CuArray{T}}, xs::SparseMatrixCSR) where {T} = cuSPARSE.CuSparseMatrixCSR{T}(xs)
Adapt.adapt_storage(::Type{Array}, mat::cuSPARSE.CuSparseMatrixCSR) = SparseMatrixCSR(mat)

end
