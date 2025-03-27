module SparseMatricesCSRExt

using CUDA
import CUDA: CUSPARSE, CuVector
using SparseMatricesCSR
import SparseMatricesCSR: SparseMatrixCSR
import Adapt

CUSPARSE.CuSparseMatrixCSR{T}(Mat::SparseMatrixCSR) where {T} =
    CUSPARSE.CuSparseMatrixCSR{T}(CuVector{Cint}(Mat.rowptr), CuVector{Cint}(Mat.colval),
        CuVector{T}(Mat.nzval), size(Mat))


SparseMatricesCSR.SparseMatrixCSR(A::CUSPARSE.CuSparseMatrixCSR) =
    SparseMatrixCSR(CUSPARSE.SparseMatrixCSC(A)) # no direct conversion (gpu_CSR -> cpu_CSC -> cpu_CSR)


Adapt.adapt_storage(::Type{CUDA.CuArray}, mat::SparseMatrixCSR) =
    CUSPARSE.CuSparseMatrixCSR(mat)

Adapt.adapt_storage(::Type{Array}, mat::CUSPARSE.CuSparseMatrixCSR) =
    SparseMatrixCSR(mat)

end
