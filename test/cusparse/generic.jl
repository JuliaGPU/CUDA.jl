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
                                            CUSPARSE.CUSPARSE_SPMV_CSR_ALG2],
                      CuSparseMatrixCOO => [CUSPARSE.CUSPARSE_SPMV_ALG_DEFAULT,
                                            # CUSPARSE.CUSPARSE_SPMV_COO_ALG2,
                                            CUSPARSE.CUSPARSE_SPMV_COO_ALG1])

    SPMM_ALGOS = Dict(CuSparseMatrixCSC => [CUSPARSE.CUSPARSE_SPMM_ALG_DEFAULT],
                      CuSparseMatrixCSR => [CUSPARSE.CUSPARSE_SPMM_ALG_DEFAULT,
                                            CUSPARSE.CUSPARSE_SPMM_CSR_ALG1,
                                            CUSPARSE.CUSPARSE_SPMM_CSR_ALG2,
                                            CUSPARSE.CUSPARSE_SPMM_CSR_ALG3],
                      CuSparseMatrixCOO => [CUSPARSE.CUSPARSE_SPMM_ALG_DEFAULT,
                                            CUSPARSE.CUSPARSE_SPMM_COO_ALG1,
                                            # CUSPARSE.CUSPARSE_SPMM_COO_ALG2,
                                            CUSPARSE.CUSPARSE_SPMM_COO_ALG3,
                                            CUSPARSE.CUSPARSE_SPMM_COO_ALG4])

    for SparseMatrixType in [CuSparseMatrixCSC, CuSparseMatrixCSR, CuSparseMatrixCOO]
        @testset "$SparseMatrixType -- mv! algo=$algo" for algo in SPMV_ALGOS[SparseMatrixType]
            @testset "mv! $T" for T in [Float32, Float64, ComplexF32, ComplexF64]
                for (transa, opa) in [('N', identity), ('T', transpose), ('C', adjoint)]
                    T <: Real && transa == 'C' && continue
                    SparseMatrixType == CuSparseMatrixCSC && T <: Complex && transa == 'C' && continue
                    A = sprand(T, 10, 10, 0.1)
                    B = rand(T, 10)
                    C = rand(T, 10)
                    dA = SparseMatrixType(A)
                    dB = CuArray(B)
                    dC = CuArray(C)

                    alpha = 1.2
                    beta = 1.3
                    mv!(transa, alpha, dA, dB, beta, dC, 'O', algo)
                    @test alpha * opa(A) * B + beta * C ≈ collect(dC)
                end
            end
        end

        @testset "$SparseMatrixType -- mm! algo=$algo" for algo in SPMM_ALGOS[SparseMatrixType]
            @testset "mm! $T" for T in [Float32, Float64, ComplexF32, ComplexF64]
                for (transa, opa) in [('N', identity), ('T', transpose), ('C', adjoint)]
                    for (transb, opb) in [('N', identity), ('T', transpose), ('C', adjoint)]
                        T <: Real && transa == 'C' && continue
                        T <: Real && transb == 'C' && continue
                        SparseMatrixType == CuSparseMatrixCSC && T <: Complex && transa == 'C' && continue
                        algo == CUSPARSE.CUSPARSE_SPMM_CSR_ALG3 && (transa != 'N' || transb != 'N') && continue
                        A = sprand(T, 10, 10, 0.1)
                        B = transb == 'N' ? rand(T, 10, 2) : rand(T, 2, 10)
                        C = rand(T, 10, 2)
                        dA = SparseMatrixType(A)
                        dB = CuArray(B)
                        dC = CuArray(C)

                        alpha = 1.2
                        beta = 1.3
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
                            T <: Real && transa == 'C' && continue

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
                                T <: Real && transa == 'C' && continue

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
