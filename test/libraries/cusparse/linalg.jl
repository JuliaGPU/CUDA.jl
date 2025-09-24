using CUDA.CUSPARSE
using LinearAlgebra, SparseArrays

m = 10
@testset "T = $T" for T in [Float32, Float64, ComplexF32, ComplexF64]
    A  = sprand(T, m, m, 0.2)
    B  = sprand(T, m, m, 0.3)
    ZA = spzeros(T, m, m)
    C  = I(div(m, 2))
    @testset "type = $typ" for typ in [CuSparseMatrixCSR, CuSparseMatrixCSC]
        dA = typ(A)
        dB = typ(B)
        dZA = typ(ZA)
        @testset "opnorm and norm" begin
            @test opnorm(A, Inf) ≈ opnorm(dA, Inf)
            @test opnorm(A, 1)   ≈ opnorm(dA, 1)
            @test_throws ArgumentError("p=2 is not supported") opnorm(dA, 2)
            @test norm(A, 0) ≈ norm(dA, 0)
            @test norm(A, 1) ≈ norm(dA, 1)
            @test norm(A, 2) ≈ norm(dA, 2)
            @test norm(A, Inf) ≈ norm(dA, Inf)
            @test norm(A, -Inf) ≈ norm(dA, -Inf)
        end
        @testset "triu tril exp" begin
            @test Array(triu(dA)) ≈ triu(A)
            @test Array(triu(dA, 1)) ≈ triu(A, 1)
            @test Array(tril(dA)) ≈ tril(A)
            @test Array(tril(dA, 1)) ≈ tril(A, 1)
            @test Array(exp(dA)) ≈ exp(collect(A))
        end
        @testset "kronecker product opa = $opa, opb = $opb" for opa in (identity, transpose, adjoint), opb in (identity, transpose, adjoint)
            if !(opa == transpose && opb == adjoint) && !(opa == adjoint && opb == transpose)
                @test collect(kron(opa(dA), opb(dB)))  ≈ kron(opa(A), opb(B))
                @test collect(kron(opa(dZA), opb(dB))) ≈ kron(opa(ZA), opb(B))
            end
        end
        @testset "kronecker product with I opa = $opa" for opa in (identity, transpose, adjoint)
            @test collect(kron(opa(dA), C)) ≈ kron(opa(A), C) 
            @test collect(kron(C, opa(dA))) ≈ kron(C, opa(A)) 
            @test collect(kron(opa(dZA), C)) ≈ kron(opa(ZA), C)
            @test collect(kron(C, opa(dZA))) ≈ kron(C, opa(ZA))
        end
    end
end

@testset "Reshape $typ (100,100) -> (20, 500) and droptol" for
    typ in [CuSparseMatrixCSR, CuSparseMatrixCSC]

    a = sprand(ComplexF32, 10, 10, 0.1)
    dims = (20, 5)
    A = typ(a)
    @test Array(reshape(A, dims)) ≈ reshape(a, dims)
    droptol!(a, 0.4)
    droptol!(A, 0.4)
    @test collect(A) ≈ a
end

@testset "Generalized dot product for $typ and $elty" for
    typ in [CuSparseMatrixCSR, CuSparseMatrixCSC], elty in [Int64, Float32, Float64, ComplexF64]

    N1 = 100*2
    N2 = 100*3
    x = rand(elty, N1)
    y = rand(elty, N2)
    A = sprand(elty, N1, N2, 1/N1)

    x2 = CuArray(x)
    y2 = CuArray(y)
    A2 = typ(A)

    @test dot(x, A, y) ≈ dot(x2, A2, y2)
    @test_throws DimensionMismatch("dimensions must match") dot(CUDA.rand(elty, N1+1), A2, y2)
end
