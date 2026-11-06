module SparseMatricesCSRExt

using CUDACore
using cuSPARSE
import cuSPARSE:
    CuSparseMatrixCSR, CuSparseMatrixCSC, CuSparseMatrixCOO, CuSparseMatrixBSR
using SparseMatricesCSR
using SparseArrays
import SparseMatricesCSR: SparseMatrixCSR
import Adapt

# CPU → GPU (with `Cint` indices, like the other constructors)
function cuSPARSE.CuSparseMatrixCSR{T,Ti}(Mat::SparseMatrixCSR{Bi}) where {T,Ti,Bi}
    # the GPU formats are one-based
    offset = 1 - Bi
    cuSPARSE.CuSparseMatrixCSR{T,Ti}(
        CuVector{Ti}(Mat.rowptr .+ offset), CuVector{Ti}(Mat.colval .+ offset),
        CuVector{T}(Mat.nzval), size(Mat))
end
cuSPARSE.CuSparseMatrixCSR{T}(Mat::SparseMatrixCSR) where {T} =
    cuSPARSE.CuSparseMatrixCSR{T,cuSPARSE.cuda_indtype(Mat)}(Mat)
cuSPARSE.CuSparseMatrixCSR(Mat::SparseMatrixCSR) = cuSPARSE.CuSparseMatrixCSR{eltype(Mat)}(Mat)
cuSPARSE.CuSparseMatrixCSC(Mat::SparseMatrixCSR) = CuSparseMatrixCSC(CuSparseMatrixCSR(Mat))
cuSPARSE.CuSparseMatrixCOO(Mat::SparseMatrixCSR) = CuSparseMatrixCOO(CuSparseMatrixCSR(Mat))
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
