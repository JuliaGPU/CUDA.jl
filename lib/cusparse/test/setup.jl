using Test
using CUDA
using cuSPARSE
using LinearAlgebra
using SparseArrays
using SparseArrays: rowvals, nonzeroinds, getcolptr

m = 25
n = 35
k = 10
p = 5
blockdim = 5
