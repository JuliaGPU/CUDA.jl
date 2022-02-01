using CUDA
using CUDA.CUSPARSE, SparseArrays


@testset "Tv=$Tv Ti=$Ti" for Tv in [Float32, Float64, ComplexF32, ComplexF64], Ti in [Int32, Int64]
    @testset "CuSparseMatrixCOO" begin
        S = sprand(Tv, 10, 10, 0.1)
        dS = CuSparseMatrixCOO{Tv, Ti}(S)

        @test collect(dS) ≈ S
    end

    @testset "CuSparseMatrixCSC" begin
        S = sprand(Tv, 10, 10, 0.1)
        dS = CuSparseMatrixCSC{Tv, Ti}(S)

        @test collect(dS) ≈ S
    end

    @testset "CuSparseMatrixCSR" begin
       S = transpose(sprand(Tv, 10, 10, 0.1))
       dS = CuSparseMatrixCSR{Tv, Ti}(S)
       @test collect(dS) ≈ S
    end
end
using Test

Tv, Ti = Float32, Int64
S = transpose(sprand(Tv, 10, 10, 0.1))
dS = CuSparseMatrixCSR{Tv, Int}(S);

CuSparseMatrixCSR{Tv, Int32}(dS)
dS2 = CuSparseMatrixCSC{Tv}(dS)


@which SparseMatrixCSC(dS2)
collect(dS) ≈ S
SparseMatrixCSC(dS)