using cuSPARSE
using LinearAlgebra, SparseArrays

@testset for elty in [Float32, Float64, ComplexF32, ComplexF64]
    m = 20
    nB = 2

    @testset "ldiv $elty $triangle" for triangle in [LowerTriangular, UnitLowerTriangular, UpperTriangular, UnitUpperTriangular]
        A  = rand(elty, m, m)
        A  = triangle in (UnitLowerTriangular, LowerTriangular) ? tril(A) : triu(A)
        A  = triangle in (UnitLowerTriangular, UnitUpperTriangular) ? A - Diagonal(A) + I : A
        A  = sparse(A)
        x  = rand(elty, m)
        y  = rand(elty, m)
        dy = CuArray(y)
        dx = CuArray(x)
        @testset "opa = $opa" for opa in (identity, transpose, adjoint)
            @testset "type = $SparseMatrixType" for SparseMatrixType in (CuSparseMatrixCOO, CuSparseMatrixCSC, CuSparseMatrixCSR, CuSparseMatrixBSR)
                SparseMatrixType == CuSparseMatrixCSC && elty <: Complex && opa == adjoint && continue
                dA = SparseMatrixType == CuSparseMatrixBSR ? CuSparseMatrixBSR(A, 1) : SparseMatrixType(A)
                @testset "ldiv! -- CuVector" begin
                    z  = rand(elty, m)
                    dz = CuArray(z)
                    ldiv!(triangle(opa(A)), z)
                    ldiv!(triangle(opa(dA)), dz)
                    @test z ≈ collect(dz)
                end
                # seems to be a library bug in CUDAs 12.0-12.2, only fp64 types are supported
                if SparseMatrixType != CuSparseMatrixBSR && (elty ∈ (Float64, ComplexF64) || v"12.2" < cuSPARSE.version())
                    @testset "ldiv! -- (CuVector, CuVector)" begin
                        z  = rand(elty, m)
                        dz = CuArray(z)
                        ldiv!(z, triangle(opa(A)), y)
                        ldiv!(dz, triangle(opa(dA)), dy)
                        @test z ≈ collect(dz)
                    end
                    @testset "\\ -- CuVector" begin
                        x  = triangle(opa(A)) \ y
                        dx = triangle(opa(dA)) \ dy
                        @test x ≈ collect(dx)
                    end
                end
                @testset "opb = $opb" for opb in [identity, transpose, adjoint]
                    elty <: Complex && opb == adjoint && continue
                    B  = opb == identity ? rand(elty, m, nB) : rand(elty, nB, m)
                    dB = CuArray(B)
                    B_bad = opb == identity ? rand(elty, m+1, nB) : rand(elty, nB, m+1)
                    dB_bad = CuArray(B_bad)
                    error_str = opb == identity ? "first dimensions of A ($m) and X ($(m+1)) must match when transxy is 'N'" : "first dimension of A ($m) must match second dimension of X ($(m+1)) when transxy is not 'N'"
                    @testset "ldiv! -- CuMatrix" begin
                        D  = copy(B)
                        dD = copy(dB)
                        ldiv!(triangle(opa(A)), opb(D))
                        ldiv!(triangle(opa(dA)), opb(dD))
                        @test B ≈ collect(dB)
                    end
                    if SparseMatrixType != CuSparseMatrixBSR && (elty ∈ (Float64, ComplexF64) || v"12.2" < cuSPARSE.version())
                        @testset "ldiv! -- (CuMatrix, CuMatrix)" begin
                            C = rand(elty, m, nB)
                            dC = CuArray(C)
                            ldiv!(C, triangle(opa(A)), opb(B))
                            ldiv!(dC, triangle(opa(dA)), opb(dB))
                            @test C ≈ collect(dC)
                        end
                        @testset "\\ -- CuMatrix" begin
                            C  = triangle(opa(A)) \ opb(B)
                            dC = triangle(opa(dA)) \ opb(dB)
                            @test C ≈ collect(dC)
                        end
                    end
                end
            end
        end
    end
end
