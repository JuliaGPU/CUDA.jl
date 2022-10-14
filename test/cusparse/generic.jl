using CUDA
using Adapt
using CUDA.CUSPARSE
using SparseArrays
using LinearAlgebra

if CUSPARSE.version() >= v"11.4.1"
    @testset "generic mv!" for T in [Float32, Float64]
        A = sprand(T, 10, 10, 0.1)
        x = rand(Complex{T}, 10)
        y = similar(x)
        dx = adapt(CuArray, x)
        dy = adapt(CuArray, y)

        dA = adapt(CuArray, A)
        mv!('N', T(1.0), dA, dx, T(0.0), dy, 'O')
        @test Array(dy) ≈ A * x

        dA = CuSparseMatrixCSR(dA)
        mv!('N', T(1.0), dA, dx, T(0.0), dy, 'O')
        @test Array(dy) ≈ A * x
    end
end

if CUSPARSE.version() >= v"11.4.1" # lower CUDA version doesn't support these algorithms

    SPMV_ALGOS = Dict(CuSparseMatrixCSC => [CUSPARSE.CUSPARSE_SPMV_ALG_DEFAULT],
                      CuSparseMatrixCSR => [CUSPARSE.CUSPARSE_SPMV_ALG_DEFAULT,
                                            CUSPARSE.CUSPARSE_SPMV_CSR_ALG1,
                                            CUSPARSE.CUSPARSE_SPMV_CSR_ALG2])

    SPMM_ALGOS = Dict(CuSparseMatrixCSC => [CUSPARSE.CUSPARSE_SPMM_ALG_DEFAULT],
                      CuSparseMatrixCSR => [CUSPARSE.CUSPARSE_SPMM_ALG_DEFAULT,
                                            CUSPARSE.CUSPARSE_SPMM_CSR_ALG1,
                                            CUSPARSE.CUSPARSE_SPMM_CSR_ALG2,
                                            CUSPARSE.CUSPARSE_SPMM_CSR_ALG3])

    if CUSPARSE.version() >= v"11.7.2"
        SPMV_ALGOS[CuSparseMatrixCOO] = [CUSPARSE.CUSPARSE_SPMV_ALG_DEFAULT,
                                         # CUSPARSE.CUSPARSE_SPMV_COO_ALG2,
                                         CUSPARSE.CUSPARSE_SPMV_COO_ALG1]

        SPMM_ALGOS[CuSparseMatrixCOO] = [CUSPARSE.CUSPARSE_SPMM_ALG_DEFAULT,
                                         CUSPARSE.CUSPARSE_SPMM_COO_ALG1,
                                         # CUSPARSE.CUSPARSE_SPMM_COO_ALG2,
                                         CUSPARSE.CUSPARSE_SPMM_COO_ALG3,
                                         CUSPARSE.CUSPARSE_SPMM_COO_ALG4]
    end

    for SparseMatrixType in keys(SPMV_ALGOS)
        @testset "$SparseMatrixType -- mv! algo=$algo" for algo in SPMV_ALGOS[SparseMatrixType]
            @testset "mv! $T" for T in [Float32, Float64, ComplexF32, ComplexF64]
                for (transa, opa) in [('N', identity), ('T', transpose), ('C', adjoint)]
                    SparseMatrixType == CuSparseMatrixCSC && T <: Complex && transa == 'C' && continue
                    A = sprand(T, 10, 10, 0.1)
                    B = rand(T, 10)
                    C = rand(T, 10)
                    dA = SparseMatrixType(A)
                    dB = CuArray(B)
                    dC = CuArray(C)

                    alpha = rand(T)
                    beta = rand(T)
                    mv!(transa, alpha, dA, dB, beta, dC, 'O', algo)
                    @test alpha * opa(A) * B + beta * C ≈ collect(dC)
                end
            end
        end
    end

    for SparseMatrixType in keys(SPMM_ALGOS)
        @testset "$SparseMatrixType -- mm! algo=$algo" for algo in SPMM_ALGOS[SparseMatrixType]
            @testset "mm! $T" for T in [Float32, Float64, ComplexF32, ComplexF64]
                for (transa, opa) in [('N', identity), ('T', transpose), ('C', adjoint)]
                    for (transb, opb) in [('N', identity), ('T', transpose), ('C', adjoint)]
                        SparseMatrixType == CuSparseMatrixCSC && T <: Complex && transa == 'C' && continue
                        algo == CUSPARSE.CUSPARSE_SPMM_CSR_ALG3 && (transa != 'N' || transb != 'N') && continue
                        A = sprand(T, 10, 10, 0.1)
                        B = transb == 'N' ? rand(T, 10, 2) : rand(T, 2, 10)
                        C = rand(T, 10, 2)
                        dA = SparseMatrixType(A)
                        dB = CuArray(B)
                        dC = CuArray(C)

                        alpha = rand(T)
                        beta = rand(T)
                        mm!(transa, transb, alpha, dA, dB, beta, dC, 'O', algo)
                        @test alpha * opa(A) * opb(B) + beta * C ≈ collect(dC)
                    end
                end
            end
        end
    end
end # version >= 11.4.1

if CUSPARSE.version() >= v"11.7.0"

    SPSV_ALGOS = Dict(CuSparseMatrixCSC => [CUSPARSE.CUSPARSE_SPSV_ALG_DEFAULT],
                      CuSparseMatrixCSR => [CUSPARSE.CUSPARSE_SPSV_ALG_DEFAULT],
                      CuSparseMatrixCOO => [CUSPARSE.CUSPARSE_SPSV_ALG_DEFAULT])

    SPSM_ALGOS = Dict(CuSparseMatrixCSC => [CUSPARSE.CUSPARSE_SPSM_ALG_DEFAULT],
                      CuSparseMatrixCSR => [CUSPARSE.CUSPARSE_SPSM_ALG_DEFAULT],
                      CuSparseMatrixCOO => [CUSPARSE.CUSPARSE_SPSM_ALG_DEFAULT])

    for SparseMatrixType in [CuSparseMatrixCSC, CuSparseMatrixCSR, CuSparseMatrixCOO]
        @testset "$SparseMatrixType -- sv! algo=$algo" for algo in SPSV_ALGOS[SparseMatrixType]
            @testset "sv! $T" for T in [Float32, Float64, ComplexF32, ComplexF64]
                for (transa, opa) in [('N', identity), ('T', transpose), ('C', adjoint)]
                    SparseMatrixType == CuSparseMatrixCSC && T <: Complex && transa == 'C' && continue
                    for uplo in ('L', 'U')
                        for diag in ('U', 'N')
                            # They forgot to conjugate the diagonal of A.
                            # It should be fixed with versions > v"11.7.3".
                            T <: Complex && transa == 'C' && diag == 'N' && continue

                            A = rand(T, 10, 10)
                            A = uplo == 'L' ? tril(A) : triu(A)
                            A = diag == 'U' ? A - Diagonal(A) + I : A
                            A = sparse(A)
                            dA = SparseMatrixType(A)

                            B = rand(T, 10)
                            C = rand(T, 10)
                            dB = CuArray(B)
                            dC = CuArray(C)
                            alpha = rand(T)
                            sv!(transa, uplo, diag, alpha, dA, dB, dC, 'O', algo)
                            @test opa(A) \ (alpha * B) ≈ collect(dC)
                        end
                    end
                end
            end
        end

        @testset "$SparseMatrixType -- mv! algo=$algo" for algo in SPSM_ALGOS[SparseMatrixType]
            @testset "mv! $T" for T in [Float32, Float64, ComplexF32, ComplexF64]
                for (transa, opa) in [('N', identity), ('T', transpose), ('C', adjoint)]
                    SparseMatrixType == CuSparseMatrixCSC && T <: Complex && transa == 'C' && continue
                    for (transb, opb) in [('N', identity), ('T', transpose)]
                        for uplo in ('L', 'U')
                            for diag in ('U', 'N')
                                # They forgot to conjugate the diagonal of A.
                                # It should be fixed with versions > v"11.7.3".
                                T <: Complex && transa == 'C' && diag == 'N' && continue

                                A = rand(T, 10, 10)
                                A = uplo == 'L' ? tril(A) : triu(A)
                                A = diag == 'U' ? A - Diagonal(A) + I : A
                                A = sparse(A)
                                dA = SparseMatrixType(A)

                                B = transb == 'N' ? rand(T, 10, 2) : rand(T, 2, 10)
                                C = rand(T, 10, 2)
                                dB = CuArray(B)
                                dC = CuArray(C)
                                alpha = rand(T)
                                sm!(transa, transb, uplo, diag, alpha, dA, dB, dC, 'O', algo)
                                @test opa(A) \ (alpha * opb(B)) ≈ collect(dC)
                            end
                        end
                    end
                end
            end
        end
    end
end # CUSPARSE.version >= 11.7.0

if CUSPARSE.version() >= v"11.3.0" # lower CUDA version doesn't support these algorithms

    fmt = Dict(CuSparseMatrixCSC => :csc,
               CuSparseMatrixCSR => :csr,
               CuSparseMatrixCOO => :coo)

    for SparseMatrixType in [CuSparseMatrixCSC, CuSparseMatrixCSR, CuSparseMatrixCOO]
        @testset "$SparseMatrixType -- densetosparse algo=$algo" for algo in [CUSPARSE.CUSPARSE_DENSETOSPARSE_ALG_DEFAULT]
            @testset "densetosparse $T" for T in [Float32, Float64, ComplexF32, ComplexF64]
                A_sparse = sprand(T, 10, 20, 0.5)
                A_dense = Matrix{T}(A_sparse)
                dA_dense = CuMatrix{T}(A_dense)
                dA_sparse = CUSPARSE.densetosparse(dA_dense, fmt[SparseMatrixType], 'O', algo)
                @test A_sparse ≈ collect(dA_sparse)
            end
        end
        @testset "$SparseMatrixType -- sparsetodense algo=$algo" for algo in [CUSPARSE.CUSPARSE_SPARSETODENSE_ALG_DEFAULT]
            @testset "sparsetodense $T" for T in [Float32, Float64, ComplexF32, ComplexF64]
                A_dense = rand(T, 10, 20)
                A_sparse = sparse(A_dense)
                dA_sparse = SparseMatrixType(A_sparse)
                dA_dense = CUSPARSE.sparsetodense(dA_sparse, 'O', algo)
                @test A_dense ≈ collect(dA_dense)
            end
        end
    end

    @testset "vv! $T" for T in [Float32, Float64, ComplexF32, ComplexF64]
        for (transx, opx) in [('N', identity), ('C', conj)]
            T <: Real && transx == 'C' && continue
            X = sprand(T, 20, 0.5)
            dX = CuSparseVector{T}(X)
            Y = rand(T, 20)
            dY = CuVector{T}(Y)
            result = vv!(transx, dX, dY, 'O')
            @test sum(opx(X[i]) * Y[i] for i=1:20) ≈ result
        end
    end

    @testset "gather! $T" for T in [Float32, Float64, ComplexF32, ComplexF64]
        X = sprand(T, 20, 0.5)
        dX = CuSparseVector{T}(X)
        Y = rand(T, 20)
        dY = CuVector{T}(Y)
        CUSPARSE.gather!(dX, dY, 'O')
        Z = copy(X)
        for i = 1:nnz(X)
            Z[X.nzind[i]] = Y[X.nzind[i]]
        end
        @test Z ≈ sparse(collect(dX))
    end

    @testset "scatter! $T" for T in [Float32, Float64, ComplexF32, ComplexF64]
        X = sprand(T, 20, 0.5)
        dX = CuSparseVector{T}(X)
        Y = rand(T, 20)
        dY = CuVector{T}(Y)
        CUSPARSE.scatter!(dY, dX, 'O')
        Z = copy(Y)
        for i = 1:nnz(X)
            Z[X.nzind[i]] = X.nzval[i]
        end
        @test Z ≈ collect(dY)
    end

    @testset "axpby! $T" for T in [Float32, Float64, ComplexF32, ComplexF64]
        X = sprand(T, 20, 0.5)
        dX = CuSparseVector{T}(X)
        Y = rand(T, 20)
        dY = CuVector{T}(Y)
        alpha = rand(T)
        beta = rand(T)
        CUSPARSE.axpby!(alpha, dX, beta, dY, 'O')
        @test alpha * X + beta * Y ≈ collect(dY)
    end

    @testset "rot! $T" for T in [Float32, Float64, ComplexF32, ComplexF64]
        X = sprand(T, 20, 0.5)
        dX = CuSparseVector{T}(X)
        Y = rand(T, 20)
        dY = CuVector{T}(Y)
        c = rand(T)
        s = rand(T)
        CUSPARSE.rot!(dX, dY, c, s, 'O')
        W = copy(X)
        Z = copy(Y)
        for i = 1:nnz(X)
            W[X.nzind[i]] =  c * X.nzval[i] + s * Y[X.nzind[i]]
            Z[X.nzind[i]] = -s * X.nzval[i] + c * Y[X.nzind[i]]
        end
        @test W ≈ collect(dX)
        @test Z ≈ collect(dY)
    end
end # CUSPARSE.version >= 11.3.0

if CUSPARSE.version() >= v"11.1.1"

    SPGEMM_ALGOS = Dict(CuSparseMatrixCSR => [CUSPARSE.CUSPARSE_SPGEMM_DEFAULT])
    if CUSPARSE.version() >= v"11.6.0"
        push!(SPGEMM_ALGOS[CuSparseMatrixCSR], CUSPARSE.CUSPARSE_SPGEMM_CSR_ALG_DETERMINITIC,
                                               CUSPARSE.CUSPARSE_SPGEMM_CSR_ALG_NONDETERMINITIC)
    end

    for SparseMatrixType in keys(SPGEMM_ALGOS)
        @testset "$SparseMatrixType -- gemm -- gemm! algo=$algo" for algo in SPGEMM_ALGOS[SparseMatrixType]
            @testset "gemm -- gemm! $T" for T in [Float32, Float64, ComplexF32, ComplexF64]
                for (transa, opa) in [('N', identity)]
                    for (transb, opb) in [('N', identity)]
                        A = sprand(T,25,10,0.2)
                        B = sprand(T,10,35,0.3)
                        dA = SparseMatrixType(A)
                        dB = SparseMatrixType(B)
                        alpha = rand(T)
                        C = alpha * opa(A) * opb(B)
                        dC = gemm(transa, transb, alpha, dA, dB, 'O', algo)
                        @test C ≈ SparseMatrixCSC(dC)

                        beta = rand(T)
                        gamma = rand(T)
                        D = gamma * opa(A) * opa(B) + beta * C

                        dD = gemm(transa, transb, gamma, dA, dB, beta, dC, 'O', algo, same_pattern=true)
                        @test D ≈ SparseMatrixCSC(dD)

                        gemm!(transa, transb, gamma, dA, dB, beta, dC, 'O', algo)
                        @test D ≈ SparseMatrixCSC(dC)

                        E = sprand(T,25,35,0.1)
                        dE = SparseMatrixType(E)
                        F = alpha * opa(A) * opb(B) + beta * E
                        dF = gemm(transa, transb, alpha, dA, dB, beta, dE, 'O', algo, same_pattern=false)
                        @test F ≈ SparseMatrixCSC(dF)
                    end
                end
            end
        end
    end
end

if CUSPARSE.version() >= v"11.4.1"

    SDDMM_ALGOS = Dict(CuSparseMatrixCSR => [CUSPARSE.CUSPARSE_SDDMM_ALG_DEFAULT])

    for SparseMatrixType in keys(SDDMM_ALGOS)
        @testset "$SparseMatrixType -- sddmm! algo=$algo" for algo in SDDMM_ALGOS[SparseMatrixType]
            @testset "sddmm! $T" for T in [Float32, Float64, ComplexF32, ComplexF64]
                for (transa, opa) in [('N', identity), ('T', transpose), ('C', adjoint)]
                    for (transb, opb) in [('N', identity), ('T', transpose), ('C', adjoint)]
                        T <: Complex && (transa == 'C' || transb == 'C') && continue
                        mA = transa == 'N' ? 25 : 10
                        nA = transa == 'N' ? 10 : 25
                        mB = transb == 'N' ? 10 : 35
                        nB = transb == 'N' ? 35 : 10

                        A = rand(T,mA,nA)
                        B = rand(T,mB,nB)
                        C = sprand(T,25,35,0.3)

                        spyC = copy(C)
                        spyC.nzval .= one(T)

                        dA = CuArray(A)
                        dB = CuArray(B)
                        dC = SparseMatrixType(C)

                        alpha = rand(T)
                        beta = rand(T)

                        D = alpha * (opa(A) * opb(B)) .* spyC + beta * C
                        sddmm!(transa, transb, alpha, dA, dB, beta, dC, 'O', algo)
                        @test collect(dC) ≈ D
                    end
                end
            end
        end
    end
end
