using Test
using CUDACore
using cuSPARSE
using LinearAlgebra
using SparseArrays
using SparseArrays: rowvals, nonzeroinds, getcolptr

m = 25
n = 35
k = 10
p = 5
blockdim = 5

# Before cuSPARSE 11.3, dense conversions use the BLAS-typed CSR/CSC routines.
dense_conversion(::Type{T}, fmt::Symbol) where {T} =
    cuSPARSE.version() >= v"11.3" ||
    (T <: LinearAlgebra.BlasFloat && fmt in (:csc, :csr, :bsr))
dense_conversion(::Type{T}, fmt::Type) where {T} =
    dense_conversion(T, fmt <: CuSparseMatrixCSC ? :csc :
                        fmt <: CuSparseMatrixCSR ? :csr :
                        fmt <: CuSparseMatrixBSR ? :bsr : :coo)
