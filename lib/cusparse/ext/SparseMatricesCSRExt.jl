module SparseMatricesCSRExt

using CUDACore
using cuSPARSE
using GPUArrays
import cuSPARSE:
    CuSparseMatrixCSR, CuSparseMatrixCSC, CuSparseMatrixCOO, CuSparseMatrixBSR
using SparseMatricesCSR
import SparseMatricesCSR: SparseMatrixCSR
import Adapt

# CPU → GPU
function cuSPARSE.CuSparseMatrixCSR{T,Ti}(Mat::SparseMatrixCSR{Bi}) where {T,Ti,Bi}
    Bi==1 || throw(ArgumentError("canonical arrays require one-based SparseMatrixCSR input"))
    GPUArrays.check_sparse_dims(Ti,size(Mat))
    GPUArrays.check_sparse_nnz(Ti,length(Mat.nzval))
    GPUArrays.check_sparse_values(T,Mat.nzval)
    cuSPARSE.CuSparseMatrixCSR(CuVector{Ti}(Mat.rowptr),CuVector{Ti}(Mat.colval),
                             CuVector{T}(Mat.nzval),size(Mat))
end
cuSPARSE.CuSparseMatrixCSR{T}(Mat::SparseMatrixCSR) where {T} = CuSparseMatrixCSR{T,Cint}(Mat)
cuSPARSE.CuSparseMatrixCSR(Mat::SparseMatrixCSR{Bi,T}) where {Bi,T} = CuSparseMatrixCSR{T}(Mat)
cuSPARSE.CuSparseMatrixCSC{T}(Mat::SparseMatrixCSR) where {T} = CuSparseMatrixCSC(CuSparseMatrixCSR{T}(Mat))
cuSPARSE.CuSparseMatrixCOO{T}(Mat::SparseMatrixCSR) where {T} = CuSparseMatrixCOO(CuSparseMatrixCSR{T}(Mat))
for format in (:CuSparseMatrixCSC,:CuSparseMatrixCOO)
    @eval cuSPARSE.$format{T,Ti}(Mat::SparseMatrixCSR) where {T,Ti} =
        $format{T,Ti}(CuSparseMatrixCSR{T,Ti}(Mat))
end
cuSPARSE.CuSparseMatrixBSR{T}(Mat::SparseMatrixCSR, blockdim) where {T} = CuSparseMatrixBSR(CuSparseMatrixCSR{T}(Mat), blockdim)
cuSPARSE.CuSparseMatrixBSR(Mat::SparseMatrixCSR,blockdim) = CuSparseMatrixBSR(CuSparseMatrixCSR(Mat),blockdim)

# GPU → CPU
function SparseMatricesCSR.SparseMatrixCSR(A::cuSPARSE.CuSparseMatrixCSR)
    Ti=promote_type(eltype(A.rowPtr),eltype(A.colVal))
    SparseMatrixCSR{1}(size(A)...,Vector{Ti}(Array(A.rowPtr)),Vector{Ti}(Array(A.colVal)),Array(A.nzVal))
end
SparseMatricesCSR.SparseMatrixCSR(A::cuSPARSE.CuSparseMatrixCOO) = SparseMatrixCSR(CuSparseMatrixCSR(A))
SparseMatricesCSR.SparseMatrixCSR(A::cuSPARSE.CuSparseMatrixCSC) = SparseMatrixCSR(CuSparseMatrixCSR(A))
SparseMatricesCSR.SparseMatrixCSR(A::cuSPARSE.CuSparseMatrixBSR) = SparseMatrixCSR(CuSparseMatrixCSR(A))

# Adapt
Adapt.adapt_storage(::Type{CuArray}, xs::SparseMatrixCSR) = cuSPARSE.CuSparseMatrixCSR(xs)
Adapt.adapt_storage(::Type{CuArray{T}}, xs::SparseMatrixCSR) where {T} = cuSPARSE.CuSparseMatrixCSR{T}(xs)
Adapt.adapt_storage(::Type{Array}, mat::cuSPARSE.CuSparseMatrixCSR) = SparseMatrixCSR(mat)

end
