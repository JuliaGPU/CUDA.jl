using CUDA.CUSPARSE
using LinearAlgebra, SparseArrays, Adapt

m = 10
@testset "T = $T" for T in [Float32, Float64, ComplexF32, ComplexF64]
    A  = sprand(T, m, m, 0.2)
    B  = sprand(T, m, m, 0.3)
    ZA = spzeros(T, m, m)
    C  = I(div(m, 2))
    D = Diagonal(rand(T, m))
    @testset "type = $typ" for typ in [CuSparseMatrixCSR, CuSparseMatrixCSC]
        dA = typ(A)
        dB = typ(B)
        dZA = typ(ZA)
        dD = adapt(CuArray, D)
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
            if CUSPARSE.version() > v"11.4.1"
                @test Array(exp(dA)) ≈ exp(collect(A))
            end
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
        @testset "kronecker product with Diagonal opa = $opa" for opa in (identity, transpose, adjoint) 
            @test collect(kron(opa(dA), dD)) ≈ kron(opa(A), D)
            @test collect(kron(dD, opa(dA))) ≈ kron(D, opa(A))
            @test collect(kron(opa(dZA), dD)) ≈ kron(opa(ZA), D)
            @test collect(kron(dD, opa(dZA))) ≈ kron(D, opa(ZA))
        end
    end
end

@testset "T = $T" for T in [Float32, Float64, ComplexF32, ComplexF64]
    mat_sizes = [(2, 3), (2, 0)]
    @testset "size(A) = ($(mA), $(nA)), size(B) = ($(mB), $(nB))" for (mA, nA) in mat_sizes, (mB, nB) in mat_sizes
        A = sprand(T, mA, nA, 0.5)
        B  = sprand(T, mB, nB, 0.5)

        A_I, A_J, A_V = findnz(A)
        dA = CuSparseMatrixCOO{T, Cint}(adapt(CuVector{Cint}, A_I), adapt(CuVector{Cint}, A_J), adapt(CuVector{T}, A_V), size(A))
        B_I, B_J, B_V = findnz(B)
        dB = CuSparseMatrixCOO{T, Cint}(adapt(CuVector{Cint}, B_I), adapt(CuVector{Cint}, B_J), adapt(CuVector{T}, B_V), size(B))

        @testset "kronecker (COO ⊗ COO) opa = $opa, opb = $opb" for opa in (identity, transpose, adjoint), opb in (identity, transpose, adjoint)
            dC = kron(opa(dA), opb(dB))
            @test collect(dC)  ≈ kron(opa(A), opb(B))
            @test eltype(dC) == typeof(oneunit(T) * oneunit(T))
            @test dC isa CuSparseMatrixCOO
        end
    end
end

@testset "TA = $TA, TvB = $TvB" for TvB in [Float32, Float64, ComplexF32, ComplexF64], TA in [Bool, TvB]
    A = Diagonal(rand(TA, 2))
    B  = sprand(TvB, 3, 4, 0.5)
    dA = adapt(CuArray, A)

    B_I, B_J, B_V = findnz(B)
    dB = CuSparseMatrixCOO{TvB, Cint}(adapt(CuVector{Cint}, B_I), adapt(CuVector{Cint}, B_J), adapt(CuVector{TvB}, B_V), size(B))

    @testset "kronecker (diagonal ⊗ COO) opa = $opa, opb = $opb" for opa in (identity, adjoint), opb in (identity, transpose, adjoint)
        dC = kron(opa(dA), opb(dB))
        @test collect(dC)  ≈ kron(opa(A), opb(B))
        @test eltype(dC) == typeof(oneunit(TA) * oneunit(TvB))
        @test dC isa CuSparseMatrixCOO
    end

    @testset "kronecker (COO ⊗ diagonal) opa = $opa, opb = $opb" for opa in (identity, adjoint), opb in (identity, transpose, adjoint)
        dC = kron(opb(dB), opa(dA))
        @test collect(dC)  ≈ kron(opb(B), opa(A))
        @test eltype(dC) == typeof(oneunit(TvB) * oneunit(TA))
        @test dC isa CuSparseMatrixCOO
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
