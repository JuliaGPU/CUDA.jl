using CUDA
using Adapt
using CUDA.CUSPARSE
using LinearAlgebra, SparseArrays

@testset "LinearAlgebra" begin
    @testset "CuSparseVector -- axpby -- A ± B $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64]
        n = 10
        alpha = rand(elty)
        beta = rand(elty)
        A = sprand(elty, n, 0.3)
        B = sprand(elty, n, 0.7)

        dA = CuSparseVector(A)
        dB = CuSparseVector(B)

        C = alpha * A + beta * B
        dC = axpby(alpha, dA, beta, dB, 'O')
        @test C ≈ collect(dC)

        C = A + B
        dC = dA + dB
        @test C ≈ collect(dC)

        C = A - B
        dC = dA - dB
        @test C ≈ collect(dC)
    end

    for SparseMatrixType in (CuSparseMatrixCSC, CuSparseMatrixCSR)
        @testset "$SparseMatrixType -- geam $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64]
            n = 10
            m = 20
            alpha = rand(elty)
            beta = rand(elty)
            A = sprand(elty, n, m, 0.2)
            B = sprand(elty, n, m, 0.5)

            dA = SparseMatrixType(A)
            dB = SparseMatrixType(B)

            C = alpha * A + beta * B
            dC = geam(alpha, dA, beta, dB, 'O')
            @test C ≈ collect(dC)
        end
    end

    for SparseMatrixType in (CuSparseMatrixCSC, CuSparseMatrixCSR, CuSparseMatrixCOO)
        @testset "$SparseMatrixType ± $SparseMatrixType -- A ± B $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64]
            for opa in (identity, transpose, adjoint)
                for opb in (identity, transpose, adjoint)
                    n = 10
                    m = 20
                    A = opa == identity ? sprand(elty, n, m, 0.2) : sprand(elty, m, n, 0.2)
                    B = opb == identity ? sprand(elty, n, m, 0.5) : sprand(elty, m, n, 0.5)

                    dA = SparseMatrixType(A)
                    dB = SparseMatrixType(B)

                    C = opa(A) + opb(B)
                    dC = opa(dA) + opb(dB)
                    @test C ≈ collect(dC)

                    C = opa(A) - opb(B)
                    dC = opa(dA) - opb(dB)
                    @test C ≈ collect(dC)
                end
            end
        end
    end

    # SpGEMM was added in CUSPARSE v"11.1.1"
    if CUSPARSE.version() >= v"11.1.1"
        for SparseMatrixType in (CuSparseMatrixCSC, CuSparseMatrixCSR, CuSparseMatrixCOO)
            @testset "$SparseMatrixType * $SparseMatrixType -- A * B $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64]
                for opa in (identity, transpose, adjoint)
                    for opb in (identity, transpose, adjoint)
                        n = 10
                        k = 15
                        m = 20
                        A = opa == identity ? sprand(elty, m, k, 0.2) : sprand(elty, k, m, 0.2)
                        B = opb == identity ? sprand(elty, k, n, 0.5) : sprand(elty, n, k, 0.5)

                        dA = SparseMatrixType(A)
                        dB = SparseMatrixType(B)

                        C = opa(A) * opb(B)
                        dC = opa(dA) * opb(dB)
                        @test C ≈ collect(dC)
                    end
                end
            end
        end
    end

    @testset "CuMatrix * CuSparseVector -- A * b $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64]
        for opa in (identity, transpose, adjoint)
            elty <: Complex && opa == adjoint && continue
            n = 10
            m = 20
            A = opa == identity ? rand(elty, n, m) : rand(elty, m, n)
            b = sprand(elty, m, 0.5)

            dA = CuArray(A)
            db = CuSparseVector(b)

            c = opa(A) * b
            dc = opa(dA) * db
            @test c ≈ collect(dc)
        end
    end

    @testset "CuMatrix * CuSparseVector -- mul!(c, A, b) $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64]
        for opa in (identity, transpose, adjoint)
            elty <: Complex && opa == adjoint && continue
            n = 10
            m = 20
            alpha = rand(elty)
            beta = rand(elty)
            A = opa == identity ? rand(elty, n, m) : rand(elty, m, n)
            b = sprand(elty, m, 0.5)
            c = rand(elty, n)

            dA = CuArray(A)
            db = CuSparseVector(b)
            dc = CuArray(c)

            mul!(c, opa(A), b, alpha, beta)
            mul!(dc, opa(dA), db, alpha, beta)
            @test c ≈ collect(dc)
        end
    end

    for SparseMatrixType in (CuSparseMatrixCSC, CuSparseMatrixCSR, CuSparseMatrixCOO)

        if CUSPARSE.version() >= v"11.7.4"
            @testset "CuMatrix * $SparseMatrixType -- A * B $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64]
                for opa in (identity, transpose, adjoint)
                    for opb in (identity, transpose, adjoint)
                        CUSPARSE.version() < v"12.0" && SparseMatrixType == CuSparseMatrixCSR && elty <: Complex && opb == adjoint && continue
                        CUSPARSE.version() ≥ v"12.0" && opb != identity && continue
                        n = 10
                        k = 15
                        m = 20
                        A = opa == identity ? rand(elty, m, k) : rand(elty, k, m)
                        B = opb == identity ? sprand(elty, k, n, 0.2) : sprand(elty, n, k, 0.2)

                        dA = CuArray(A)
                        dB = SparseMatrixType(B)
                        if SparseMatrixType == CuSparseMatrixCOO
                            dB = sort_coo(dB, 'C')
                        end

                        C = opa(A) * opb(B)
                        dC = opa(dA) * opb(dB)
                        @test C ≈ collect(dC)
                    end
                end
            end

            @testset "CuMatrix * $SparseMatrixType -- mul!(C, A, B) $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64]
                for opa in (identity, transpose, adjoint)
                    for opb in (identity, transpose, adjoint)
                        CUSPARSE.version() < v"12.0" && SparseMatrixType == CuSparseMatrixCSR && elty <: Complex && opb == adjoint && continue
                        CUSPARSE.version() ≥ v"12.0" && opb != identity && continue
                        n = 10
                        k = 15
                        m = 20
                        alpha = rand(elty)
                        beta = rand(elty)
                        A = opa == identity ? rand(elty, m, k) : rand(elty, k, m)
                        B = opb == identity ? sprand(elty, k, n, 0.2) : sprand(elty, n, k, 0.2)
                        C = rand(elty, m, n)

                        dA = CuArray(A)
                        dB = SparseMatrixType(B)
                        if SparseMatrixType == CuSparseMatrixCOO
                            dB = sort_coo(dB, 'C')
                        end
                        dC = CuArray(C)

                        mul!(C, opa(A), opb(B), alpha, beta)
                        mul!(dC, opa(dA), opb(dB), alpha, beta)
                        @test C ≈ collect(dC)
                    end
                end
            end
        end

        @testset "$SparseMatrixType * CuVector -- A * b $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64]
            for opa in (identity, transpose, adjoint)
                CUSPARSE.version() < v"12.0" && SparseMatrixType == CuSparseMatrixCSC && elty <: Complex && opa == adjoint && continue
                n = 10
                m = 20
                A = opa == identity ? sprand(elty, n, m, 0.2) : sprand(elty, m, n, 0.2)
                b = rand(elty, m)

                dA = SparseMatrixType(A)
                db = CuArray(b)

                c = opa(A) * b
                dc = opa(dA) * db
                @test c ≈ collect(dc)
            end
        end

        @testset "$SparseMatrixType * CuVector -- mul!(c, A, b) $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64]
            for opa in (identity, transpose, adjoint)
                CUSPARSE.version() < v"12.0" && SparseMatrixType == CuSparseMatrixCSC && elty <: Complex && opa == adjoint && continue
                n = 10
                m = 20
                alpha = rand(elty)
                beta = rand(elty)
                A = opa == identity ? sprand(elty, n, m, 0.2) : sprand(elty, m, n, 0.2)
                b = rand(elty, m)
                c = rand(elty, n)

                dA = SparseMatrixType(A)
                db = CuArray(b)
                dc = CuArray(c)

                mul!(c, opa(A), b, alpha, beta)
                mul!(dc, opa(dA), db, alpha, beta)
                @test c ≈ collect(dc)
            end
        end

        @testset "$SparseMatrixType * CuMatrix -- A * B $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64]
            for opa in (identity, transpose, adjoint)
                for opb in (identity, transpose, adjoint)
                    CUSPARSE.version() < v"12.0" && SparseMatrixType == CuSparseMatrixCSC && elty <: Complex && opa == adjoint && continue
                    n = 10
                    k = 15
                    m = 20
                    A = opa == identity ? sprand(elty, m, k, 0.2) : sprand(elty, k, m, 0.2)
                    B = opb == identity ? rand(elty, k, n) : rand(elty, n, k)

                    dA = SparseMatrixType(A)
                    dB = CuArray(B)

                    C = opa(A) * opb(B)
                    dC = opa(dA) * opb(dB)
                    @test C ≈ collect(dC)
                end
            end
        end

        @testset "$SparseMatrixType * CuMatrix -- mul!(C, A, B) $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64]
            for opa in (identity, transpose, adjoint)
                for opb in (identity, transpose, adjoint)
                    CUSPARSE.version() < v"12.0" && SparseMatrixType == CuSparseMatrixCSC && elty <: Complex && opa == adjoint && continue
                    n = 10
                    k = 15
                    m = 20
                    alpha = rand(elty)
                    beta = rand(elty)
                    A = opa == identity ? sprand(elty, m, k, 0.2) : sprand(elty, k, m, 0.2)
                    B = opb == identity ? rand(elty, k, n) : rand(elty, n, k)
                    C = rand(elty, m, n)

                    dA = SparseMatrixType(A)
                    dB = CuArray(B)
                    dC = CuArray(C)

                    mul!(C, opa(A), opb(B), alpha, beta)
                    mul!(dC, opa(dA), opb(dB), alpha, beta)
                    @test C ≈ collect(dC)
                end
            end
        end

        @testset "$SparseMatrixType * CuSparseVector -- A * b $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64]
            for opa in (identity, transpose, adjoint)
                CUSPARSE.version() < v"12.0" && SparseMatrixType == CuSparseMatrixCSC && elty <: Complex && opa == adjoint && continue
                n = 10
                m = 20
                A = opa == identity ? sprand(elty, n, m, 0.2) : sprand(elty, m, n, 0.2)
                b = sprand(elty, m, 0.5)

                dA = SparseMatrixType(A)
                db = CuSparseVector(b)

                c = opa(A) * b
                dc = opa(dA) * db
                @test c ≈ collect(dc)
            end
        end

        @testset "$SparseMatrixType * CuSparseVector -- mul!(c, A, b) $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64]
            for opa in (identity, transpose, adjoint)
                CUSPARSE.version() < v"12.0" && SparseMatrixType == CuSparseMatrixCSC && elty <: Complex && opa == adjoint && continue
                n = 10
                m = 20
                alpha = rand(elty)
                beta = rand(elty)
                A = opa == identity ? sprand(elty, n, m, 0.2) : sprand(elty, m, n, 0.2)
                b = sprand(elty, m, 0.5)
                c = rand(elty, n)

                dA = SparseMatrixType(A)
                db = CuSparseVector(b)
                dc = CuArray(c)

                mul!(c, opa(A), b, alpha, beta)
                mul!(dc, opa(dA), db, alpha, beta)
                @test c ≈ collect(dc)
            end
        end
    end

    @testset "CuSparseMatrixCSR($f) $elty" for f in [transpose, adjoint], elty in [Float32, ComplexF32]
        S = f(sprand(elty, 10, 10, 0.1))
        @test SparseMatrixCSC(CuSparseMatrixCSR(S)) ≈ S

        S = sprand(elty, 10, 10, 0.1)
        T = f(CuSparseMatrixCSR(S))
        @test SparseMatrixCSC(CuSparseMatrixCSC(T)) ≈ f(S)

        S = sprand(elty, 10, 10, 0.1)
        T = f(CuSparseMatrixCSC(S))
        @test SparseMatrixCSC(CuSparseMatrixCSR(T)) ≈ f(S)
    end

    VERSION >= v"1.7" && @testset "UniformScaling with $typ($dims)" for
            typ in [CuSparseMatrixCSR, CuSparseMatrixCSC],
            dims in [(10, 10), (5, 10), (10, 5)]
        S = sprand(Float32, dims..., 0.1)
        dA = typ(S)

        @test Array(dA + I) == S + I
        @test Array(I + dA) == I + S

        @test Array(dA - I) == S - I
        @test Array(I - dA) == I - S
    end

    @testset "Diagonal with $typ(10, 10)" for
        typ in [CuSparseMatrixCSR, CuSparseMatrixCSC]

        S = sprand(Float32, 10, 10, 0.8)
        D = Diagonal(rand(Float32, 10))
        dA = typ(S)
        dD = adapt(CuArray, D)

        @test Array(dA + dD) == S + D
        @test Array(dD + dA) == D + S

        @test Array(dA - dD) == S - D
        @test Array(dD - dA) == D - S

        @test dA + dD isa typ
        @test dD + dA isa typ
        @test dA - dD isa typ
        @test dD - dA isa typ
    end

    @testset "dot(CuVector, CuSparseVector) and dot(CuSparseVector, CuVector) $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64],
        n = 10
        x = sprand(elty, n, 0.5)
        y = rand(elty, n)

        dx = CuSparseVector(x)
        dy = CuVector(y)

        @test dot(dx,dy) ≈ dot(x,y)
        @test dot(dy,dx) ≈ dot(y,x)
    end

    for triangle in [LowerTriangular, UnitLowerTriangular, UpperTriangular, UnitUpperTriangular]
        @testset "ldiv!($triangle(CuSparseMatrixBSR), CuVector) -- $elty" for elty in [Float32,Float64,ComplexF32,ComplexF64]
            for opa in [identity, transpose, adjoint]
                A = A = rand(elty, 10, 10)
                A = triangle in (UnitLowerTriangular, LowerTriangular) ? tril(A) : triu(A)
                A = triangle in (UnitLowerTriangular, UnitUpperTriangular) ? A - Diagonal(A) + I : A
                A = sparse(A)
                y = rand(elty, 10)
                dA = CuSparseMatrixBSR(A, 1)
                dy = CuArray(y)
                ldiv!(triangle(opa(A)), y)
                ldiv!(triangle(opa(dA)), dy)
                @test y ≈ collect(dy)
             end
        end

        @testset "$triangle(CuSparseMatrixBSR) \\ CuVector -- $elty" for elty in [Float32,Float64,ComplexF32,ComplexF64]
            for opa in [identity, transpose, adjoint]
                A = A = rand(elty, 10, 10)
                A = triangle in (UnitLowerTriangular, LowerTriangular) ? tril(A) : triu(A)
                A = triangle in (UnitLowerTriangular, UnitUpperTriangular) ? A - Diagonal(A) + I : A
                A = sparse(A)
                y = rand(elty, 10)
                dA = CuSparseMatrixBSR(A, 1)
                dy = CuArray(y)
                x = triangle(opa(A)) \ y
                dx = triangle(opa(dA)) \ dy
                @test x ≈ collect(dx)
             end
        end

        @testset "ldiv!($triangle(CuSparseMatrixBSR), CuMatrix) -- $elty" for elty in [Float32,Float64,ComplexF32,ComplexF64]
            for opa in [identity, transpose, adjoint]
                A = A = rand(elty, 10, 10)
                A = triangle in (UnitLowerTriangular, LowerTriangular) ? tril(A) : triu(A)
                A = triangle in (UnitLowerTriangular, UnitUpperTriangular) ? A - Diagonal(A) + I : A
                A = sparse(A)
                B = rand(elty, 10, 2)
                dA = CuSparseMatrixBSR(A, 1)
                dB = CuArray(B)
                ldiv!(triangle(opa(A)), B)
                ldiv!(triangle(opa(dA)), dB)
                @test B ≈ collect(dB)
            end
        end

        @testset "$triangle(CuSparseMatrixBSR) \\ CuMatrix -- $elty" for elty in [Float32,Float64,ComplexF32,ComplexF64]
            for opa in [identity, transpose, adjoint]
                A = A = rand(elty, 10, 10)
                A = triangle in (UnitLowerTriangular, LowerTriangular) ? tril(A) : triu(A)
                A = triangle in (UnitLowerTriangular, UnitUpperTriangular) ? A - Diagonal(A) + I : A
                A = sparse(A)
                B = rand(elty, 10, 2)
                dA = CuSparseMatrixBSR(A, 1)
                dB = CuArray(B)
                C = triangle(opa(A)) \ B
                dC = triangle(opa(dA)) \ dB
                @test C ≈ collect(dC)
            end
        end
    end

    for SparseMatrixType in (CuSparseMatrixCOO, CuSparseMatrixCSC, CuSparseMatrixCSR)
        SparseMatrixType == CuSparseMatrixCOO && CUSPARSE.version() < v"12.0" && continue
        for triangle in [LowerTriangular, UnitLowerTriangular, UpperTriangular, UnitUpperTriangular]
            @testset "ldiv!(CuVector, $triangle($SparseMatrixType), CuVector) -- $elty" for elty in [Float64,ComplexF64]
                for opa in [identity, transpose, adjoint]
                    SparseMatrixType == CuSparseMatrixCSC && elty <: Complex && opa == adjoint && continue
                    A = A = rand(elty, 10, 10)
                    A = triangle in (UnitLowerTriangular, LowerTriangular) ? tril(A) : triu(A)
                    A = triangle in (UnitLowerTriangular, UnitUpperTriangular) ? A - Diagonal(A) + I : A
                    A = sparse(A)
                    y = rand(elty, 10)
                    x = rand(elty, 10)
                    dA = SparseMatrixType(A)
                    dy = CuArray(y)
                    dx = CuArray(x)
                    ldiv!(x, triangle(opa(A)), y)
                    ldiv!(dx, triangle(opa(dA)), dy)
                    @test x ≈ collect(dx)
                 end
            end

            @testset "$triangle($SparseMatrixType) \\ CuVector -- $elty" for elty in [Float64,ComplexF64]
                for opa in [identity, transpose, adjoint]
                    SparseMatrixType == CuSparseMatrixCSC && elty <: Complex && opa == adjoint && continue
                    A = rand(elty, 10, 10)
                    A = triangle in (UnitLowerTriangular, LowerTriangular) ? tril(A) : triu(A)
                    A = triangle in (UnitLowerTriangular, UnitUpperTriangular) ? A - Diagonal(A) + I : A
                    A = sparse(A)
                    y = rand(elty, 10)
                    dA = SparseMatrixType(A)
                    dy = CuArray(y)
                    x = triangle(opa(A)) \ y
                    dx = triangle(opa(dA)) \ dy
                    @test x ≈ collect(dx)
                 end
            end

            @testset "ldiv!(CuMatrix, $triangle($SparseMatrixType), CuMatrix) -- $elty" for elty in [Float64,ComplexF64]
                for opa in [identity, transpose, adjoint]
                    for opb in [identity, transpose, adjoint]
                        SparseMatrixType == CuSparseMatrixCSC && elty <: Complex && opa == adjoint && continue
                        opb != identity && CUSPARSE.version() < v"12.0" && continue
                        elty <: Complex && opb == adjoint && continue
                        A = rand(elty, 10, 10)
                        A = triangle in (UnitLowerTriangular, LowerTriangular) ? tril(A) : triu(A)
                        A = triangle in (UnitLowerTriangular, UnitUpperTriangular) ? A - Diagonal(A) + I : A
                        A = sparse(A)
                        B = opb == identity ? rand(elty, 10, 2) : rand(elty, 2, 10)
                        C = rand(elty, 10, 2)
                        dA = SparseMatrixType(A)
                        dB = CuArray(B)
                        dC = CuArray(C)
                        ldiv!(C, triangle(opa(A)), opb(B))
                        ldiv!(dC, triangle(opa(dA)), opb(dB))
                        @test C ≈ collect(dC)
                    end
                end
            end

            @testset "$triangle($SparseMatrixType) \\ CuMatrix -- $elty" for elty in [Float64,ComplexF64]
                for opa in [identity, transpose, adjoint]
                    for opb in [identity, transpose, adjoint]
                        SparseMatrixType == CuSparseMatrixCSC && elty <: Complex && opa == adjoint && continue
                        opb != identity && CUSPARSE.version() < v"12.0" && continue
                        elty <: Complex && opb == adjoint && continue
                        A = rand(elty, 10, 10)
                        A = triangle in (UnitLowerTriangular, LowerTriangular) ? tril(A) : triu(A)
                        A = triangle in (UnitLowerTriangular, UnitUpperTriangular) ? A - Diagonal(A) + I : A
                        A = sparse(A)
                        B = opb == identity ? rand(elty, 10, 2) : rand(elty, 2, 10)
                        dA = SparseMatrixType(A)
                        dB = CuArray(B)
                        C = triangle(opa(A)) \ opb(B)
                        dC = triangle(opa(dA)) \ opb(dB)
                        @test C ≈ collect(dC)
                    end
                end
            end
        end
    end
end
