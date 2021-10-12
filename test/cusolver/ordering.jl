using CUDA.CUSOLVER
using CUDA.CUSPARSE

using LinearAlgebra
using SparseArrays

n = 20
A = sparse(rand(n,n))
A = CuSparseMatrixCSR(A)

@testset "symrcm" begin
    p = symamd(A, 'Z')
    p = symamd(A, 'O')
end

@testset "symmdq" begin
    p = symmdq(A, 'Z')
    p = symmdq(A, 'O')
end

@testset "symamd" begin
    p = symamd(A, 'Z')
    p = symamd(A, 'O')
end

@testset "metisnd" begin
    p = metisnd(A, 'Z')
    p = metisnd(A, 'O')
end

@testset for elty in [Float32, Float64, ComplexF32, ComplexF64]
    @testset "zfd" begin
        A = sparse(rand(elty,n,n))
        A = CuSparseMatrixCSR(A)
        p = zfd(A, 'Z')
        P = zfd(A, 'O')
    end
end
