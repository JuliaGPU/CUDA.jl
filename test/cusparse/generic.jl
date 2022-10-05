using CUDA
using Adapt
using CUDA.CUSPARSE
using SparseArrays

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

if CUSPARSE.version() >= v"11.1.1" # lower CUDA version doesn't support these algorithms

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
end # CUSPARSE.version >= 11.1.1

if CUSPARSE.version() >= v"11.0" # lower CUDA version doesn't support these algorithms
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
end # CUSPARSE.version >= 11.0
