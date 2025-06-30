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
                    if opa == opb == identity
                        dA = SparseMatrixType(A)
                        dB = SparseMatrixType(B)
                        mul!(dC, opa(dA), opb(dB), 3, 3.2)
                        C = 3.2 * C + 3 * opa(A) * opb(B)
                        @test collect(dC) ≈ C
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

    @testset "CuSparseMatrix * CuVector -- mul!(c, A, b) mixed $eltys" for eltys in ((Float32, ComplexF32), (Float64, ComplexF64))
        eltya, eltyb = eltys
        for opa in (identity, transpose, adjoint)
            n = 10
            m = 20
            A = opa == identity ? sprand(eltya, n, m, 0.5) : sprand(eltya, m, n, 0.5)
            b = rand(eltyb, m)
            c = rand(eltyb, n)

            dA = CuSparseMatrixCSR(A)
            db = CuArray(b)
            dc = CuArray(c)

            mul!(c, opa(A), b)
            mul!(dc, opa(dA), db)
            @test c ≈ collect(dc)
        end
    end

    for SparseMatrixType in (CuSparseMatrixCSC, CuSparseMatrixCSR, CuSparseMatrixCOO, CuSparseMatrixBSR)

        if CUSPARSE.version() >= v"11.7.4"
            @testset "CuMatrix * $SparseMatrixType -- A * B $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64]
                (SparseMatrixType == CuSparseMatrixBSR) && continue
                for opa in (identity, transpose, adjoint)
                    for opb in (identity, transpose, adjoint)
                        CUSPARSE.version() < v"12.0" && SparseMatrixType == CuSparseMatrixCSR && elty <: Complex && opb == adjoint && continue
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
                (SparseMatrixType == CuSparseMatrixBSR) && continue
                for opa in (identity, transpose, adjoint)
                    for opb in (identity, transpose, adjoint)
                        CUSPARSE.version() < v"12.0" && SparseMatrixType == CuSparseMatrixCSR && elty <: Complex && opb == adjoint && continue
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
            (SparseMatrixType == CuSparseMatrixBSR) && continue
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
            (SparseMatrixType == CuSparseMatrixBSR) && continue
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
            (CUSPARSE.version() < v"12.5.1") && (SparseMatrixType == CuSparseMatrixBSR) && continue
            @testset "opa = $(string(opa))" for opa in (identity, transpose, adjoint)
                (opa != identity) && (SparseMatrixType == CuSparseMatrixBSR) && continue
                @testset "opb = $(string(opb))" for opb in (identity, transpose, adjoint)
                    CUSPARSE.version() < v"12.0" && SparseMatrixType == CuSparseMatrixCSC && elty <: Complex && opa == adjoint && continue
                    n = 10
                    k = 15
                    m = 20
                    A = opa == identity ? sprand(elty, m, k, 0.2) : sprand(elty, k, m, 0.2)
                    B = opb == identity ? rand(elty, k, n) : rand(elty, n, k)

                    dA = (SparseMatrixType == CuSparseMatrixBSR) ? SparseMatrixType(A,1) : SparseMatrixType(A)
                    dB = CuArray(B)

                    C = opa(A) * opb(B)
                    dC = opa(dA) * opb(dB)
                    @test C ≈ collect(dC)
                end
            end
        end

        @testset "$SparseMatrixType * CuMatrix -- mul!(C, A, B) $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64]
            (CUSPARSE.version() < v"12.5.1") && (SparseMatrixType == CuSparseMatrixBSR) && continue
            @testset "opa = $(string(opa))" for opa in (identity, transpose, adjoint)
                (opa != identity) && (SparseMatrixType == CuSparseMatrixBSR) && continue
                @testset "opb = $(string(opb))" for opb in (identity, transpose, adjoint)
                    CUSPARSE.version() < v"12.0" && SparseMatrixType == CuSparseMatrixCSC && elty <: Complex && opa == adjoint && continue
                    n = 10
                    k = 15
                    m = 20
                    alpha = rand(elty)
                    beta = rand(elty)
                    A = opa == identity ? sprand(elty, m, k, 0.2) : sprand(elty, k, m, 0.2)
                    B = opb == identity ? rand(elty, k, n) : rand(elty, n, k)
                    C = rand(elty, m, n)

                    dA = (SparseMatrixType == CuSparseMatrixBSR) ? SparseMatrixType(A,1) : SparseMatrixType(A)
                    dB = CuArray(B)
                    dC = CuArray(C)

                    mul!(C, opa(A), opb(B), alpha, beta)
                    mul!(dC, opa(dA), opb(dB), alpha, beta)
                    @test C ≈ collect(dC)
                end
            end
        end

        @testset "$SparseMatrixType * CuSparseVector -- A * b $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64]
            (SparseMatrixType == CuSparseMatrixBSR) && continue
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
            (SparseMatrixType == CuSparseMatrixBSR) && continue
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
        m = 10
        S = f(sprand(elty, m, m, 0.1))
        @test SparseMatrixCSC(CuSparseMatrixCSR(S)) ≈ S

        S = sprand(elty, m, m, 0.1)
        T = f(CuSparseMatrixCSR(S))
        @test SparseMatrixCSC(CuSparseMatrixCSC(T)) ≈ f(S)

        S = sprand(elty, m, m, 0.1)
        T = f(CuSparseMatrixCSC(S))
        @test SparseMatrixCSC(CuSparseMatrixCSR(T)) ≈ f(S)
    end

    @testset "UniformScaling basic operations" begin
        for elty in (Float32, Float64, ComplexF32, ComplexF64)
            m = 100
            A = sprand(elty, m, m, 0.1)
            U1 = 2*I
            for SparseMatrixType in (CuSparseMatrixCSC, CuSparseMatrixCSR, CuSparseMatrixCOO)
                B = SparseMatrixType(A)
                for op in (+, -, *)
                    @test Array(op(B, U1)) ≈ op(A, U1) && Array(op(U1, B)) ≈ op(U1, A)
                end
            end
        end
    end

    @testset "Diagonal basic operations" begin
        for elty in (Float32, Float64, ComplexF32, ComplexF64)
            m = 100
            A = sprand(elty, m, m, 0.1)
            U2 = 2*I(m)
            U3 = Diagonal(rand(elty, m))
            for SparseMatrixType in (CuSparseMatrixCSC, CuSparseMatrixCSR, CuSparseMatrixCOO)
                B = SparseMatrixType(A)
                for op in (+, -, *)
                    @test Array(op(B, U2)) ≈ op(A, U2) && Array(op(U2, B)) ≈ op(U2, A)
                    @test Array(op(B, U3)) ≈ op(A, U3) && Array(op(U3, B)) ≈ op(U3, A)
                end
            end
        end
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
            @testset "opa = $opa" for opa in [identity, transpose, adjoint]
                m = 10
                A = rand(elty, m, m)
                A = triangle in (UnitLowerTriangular, LowerTriangular) ? tril(A) : triu(A)
                A = triangle in (UnitLowerTriangular, UnitUpperTriangular) ? A - Diagonal(A) + I : A
                A = sparse(A)
                y = rand(elty, m)
                dA = CuSparseMatrixBSR(A, 1)
                dy = CuArray(y)
                ldiv!(triangle(opa(A)), y)
                ldiv!(triangle(opa(dA)), dy)
                @test y ≈ collect(dy)
             end
        end

        @testset "$triangle(CuSparseMatrixBSR) \\ CuVector -- $elty" for elty in [Float32,Float64,ComplexF32,ComplexF64]
            @testset "opa = $opa" for opa in [identity, transpose, adjoint]
                m = 10
                A = rand(elty, m, m)
                A = triangle in (UnitLowerTriangular, LowerTriangular) ? tril(A) : triu(A)
                A = triangle in (UnitLowerTriangular, UnitUpperTriangular) ? A - Diagonal(A) + I : A
                A = sparse(A)
                y = rand(elty, m)
                dA = CuSparseMatrixBSR(A, 1)
                dy = CuArray(y)
                x = triangle(opa(A)) \ y
                dx = triangle(opa(dA)) \ dy
                @test x ≈ collect(dx)
             end
        end

        @testset "ldiv!($triangle(CuSparseMatrixBSR), CuMatrix) -- $elty" for elty in [Float32,Float64,ComplexF32,ComplexF64]
            @testset "opa = $opa" for opa in [identity, transpose, adjoint]
                @testset "opb = $opb" for opb in [identity, transpose, adjoint]
                    elty <: Complex && opb == adjoint && continue
                    m = 10
                    n = 2
                    A = rand(elty, m, m)
                    A = triangle in (UnitLowerTriangular, LowerTriangular) ? tril(A) : triu(A)
                    A = triangle in (UnitLowerTriangular, UnitUpperTriangular) ? A - Diagonal(A) + I : A
                    A = sparse(A)
                    B = opb == identity ? rand(elty, m, n) : rand(elty, n, m)
                    dA = CuSparseMatrixBSR(A, 1)
                    dB = CuArray(B)
                    ldiv!(triangle(opa(A)), opb(B))
                    ldiv!(triangle(opa(dA)), opb(dB))
                    @test B ≈ collect(dB)
                    if CUSPARSE.version() < v"12.0"
                        B_bad     = opb == identity ? rand(elty, m+1, n) : rand(elty, n, m+1)
                        error_str = opb == identity ? "first dimensions of A ($m) and X ($(m+1)) must match when transxy is 'N'" : "first dimension of A ($m) must match second dimension of X ($(m+1)) when transxy is not 'N'"
                        @test_throws DimensionMismatch(error_str) ldiv!(triangle(opa(dA)), opb(CuArray(B_bad)))
                    end
                end
            end
        end

        @testset "$triangle(CuSparseMatrixBSR) \\ CuMatrix -- $elty" for elty in [Float32,Float64,ComplexF32,ComplexF64]
            @testset "opa = $opa" for opa in [identity, transpose, adjoint]
                @testset "opb = $opb" for opb in [identity, transpose, adjoint]
                    elty <: Complex && opb == adjoint && continue
                    m = 10
                    n = 2
                    A = rand(elty, m, m)
                    A = triangle in (UnitLowerTriangular, LowerTriangular) ? tril(A) : triu(A)
                    A = triangle in (UnitLowerTriangular, UnitUpperTriangular) ? A - Diagonal(A) + I : A
                    A = sparse(A)
                    B = opb == identity ? rand(elty, m, n) : rand(elty, n, m)
                    dA = CuSparseMatrixBSR(A, 1)
                    dB = CuArray(B)
                    C = triangle(opa(A)) \ opb(B)
                    dC = triangle(opa(dA)) \ opb(dB)
                    @test C ≈ collect(dC)
                    if CUSPARSE.version() < v"12.0"
                        B_bad = opb == identity ? rand(elty, m+1, n) : rand(elty, n, m+1)
                        error_str = opb == identity ? "first dimensions of A ($m) and X ($(m+1)) must match when transxy is 'N'" : "first dimension of A ($m) must match second dimension of X ($(m+1)) when transxy is not 'N'"
                        @test_throws DimensionMismatch(error_str) ldiv!(triangle(opa(dA)), opb(CuArray(B_bad)))
                    end
                end
            end
        end
    end

    for SparseMatrixType in (CuSparseMatrixCOO, CuSparseMatrixCSC, CuSparseMatrixCSR)
        SparseMatrixType == CuSparseMatrixCOO && CUSPARSE.version() < v"12.0" && continue
        for triangle in [LowerTriangular, UnitLowerTriangular, UpperTriangular, UnitUpperTriangular]
            @testset "ldiv!($triangle($SparseMatrixType), CuVector) -- $elty" for elty in [Float64,ComplexF64]
                @testset "opa = $opa" for opa in [identity, transpose, adjoint]
                    SparseMatrixType == CuSparseMatrixCSC && elty <: Complex && opa == adjoint && continue
                    m = 10
                    A = A = rand(elty, m, m)
                    A = triangle in (UnitLowerTriangular, LowerTriangular) ? tril(A) : triu(A)
                    A = triangle in (UnitLowerTriangular, UnitUpperTriangular) ? A - Diagonal(A) + I : A
                    A = sparse(A)
                    y = rand(elty, m)
                    dA = SparseMatrixType(A)
                    dy = CuArray(y)
                    ldiv!(triangle(opa(A)), y)
                    ldiv!(triangle(opa(dA)), dy)
                    @test y ≈ collect(dy)
                 end
            end

            @testset "ldiv!(CuVector, $triangle($SparseMatrixType), CuVector) -- $elty" for elty in [Float64,ComplexF64]
                @testset "opa = $opa" for opa in [identity, transpose, adjoint]
                    SparseMatrixType == CuSparseMatrixCSC && elty <: Complex && opa == adjoint && continue
                    m = 10
                    A = A = rand(elty, m, m)
                    A = triangle in (UnitLowerTriangular, LowerTriangular) ? tril(A) : triu(A)
                    A = triangle in (UnitLowerTriangular, UnitUpperTriangular) ? A - Diagonal(A) + I : A
                    A = sparse(A)
                    y = rand(elty, m)
                    x = rand(elty, m)
                    dA = SparseMatrixType(A)
                    dy = CuArray(y)
                    dx = CuArray(x)
                    ldiv!(x, triangle(opa(A)), y)
                    ldiv!(dx, triangle(opa(dA)), dy)
                    @test x ≈ collect(dx)
                 end
            end

            @testset "$triangle($SparseMatrixType) \\ CuVector -- $elty" for elty in [Float64,ComplexF64]
                @testset "opa = $opa" for opa in [identity, transpose, adjoint]
                    SparseMatrixType == CuSparseMatrixCSC && elty <: Complex && opa == adjoint && continue
                    m = 10
                    A = rand(elty, m, m)
                    A = triangle in (UnitLowerTriangular, LowerTriangular) ? tril(A) : triu(A)
                    A = triangle in (UnitLowerTriangular, UnitUpperTriangular) ? A - Diagonal(A) + I : A
                    A = sparse(A)
                    y = rand(elty, m)
                    dA = SparseMatrixType(A)
                    dy = CuArray(y)
                    x = triangle(opa(A)) \ y
                    dx = triangle(opa(dA)) \ dy
                    @test x ≈ collect(dx)
                 end
            end

            @testset "ldiv!($triangle($SparseMatrixType), CuMatrix) -- $elty" for elty in [Float64,ComplexF64]
                @testset "opa = $opa" for opa in [identity, transpose, adjoint]
                    @testset "opb = $opb" for opb in [identity, transpose, adjoint]
                        SparseMatrixType == CuSparseMatrixCSC && elty <: Complex && opa == adjoint && continue
                        elty <: Complex && opb == adjoint && continue
                        m = 10
                        n = 2
                        A = rand(elty, m, m)
                        A = triangle in (UnitLowerTriangular, LowerTriangular) ? tril(A) : triu(A)
                        A = triangle in (UnitLowerTriangular, UnitUpperTriangular) ? A - Diagonal(A) + I : A
                        A = sparse(A)
                        B = opb == identity ? rand(elty, m, n) : rand(elty, n, m)
                        dA = SparseMatrixType(A)
                        dB = CuArray(B)
                        ldiv!(triangle(opa(A)), opb(B))
                        ldiv!(triangle(opa(dA)), opb(dB))
                        @test B ≈ collect(dB)
                        if CUSPARSE.version() < v"12.0"
                            B_bad = opb == identity ? rand(elty, m+1, n) : rand(elty, n, m+1)
                            error_str = opb == identity ? "first dimensions of A ($m) and X ($(m+1)) must match when transxy is 'N'" : "first dimension of A ($m) must match second dimension of X ($(m+1)) when transxy is not 'N'"
                            @test_throws DimensionMismatch(error_str) ldiv!(triangle(opa(dA)), opb(CuArray(B_bad)))
                        end
                    end
                end
            end

            @testset "ldiv!(CuMatrix, $triangle($SparseMatrixType), CuMatrix) -- $elty" for elty in [Float64,ComplexF64]
                @testset "opa = $opa" for opa in [identity, transpose, adjoint]
                    @testset "opb = $opb" for opb in [identity, transpose, adjoint]
                        SparseMatrixType == CuSparseMatrixCSC && elty <: Complex && opa == adjoint && continue
                        elty <: Complex && opb == adjoint && continue
                        m = 10
                        n = 2
                        A = rand(elty, m, m)
                        A = triangle in (UnitLowerTriangular, LowerTriangular) ? tril(A) : triu(A)
                        A = triangle in (UnitLowerTriangular, UnitUpperTriangular) ? A - Diagonal(A) + I : A
                        A = sparse(A)
                        B = opb == identity ? rand(elty, m, n) : rand(elty, n, m)
                        C = rand(elty, m, n)
                        dA = SparseMatrixType(A)
                        dB = CuArray(B)
                        dC = CuArray(C)
                        ldiv!(C, triangle(opa(A)), opb(B))
                        ldiv!(dC, triangle(opa(dA)), opb(dB))
                        @test C ≈ collect(dC)
                        if CUSPARSE.version() < v"12.0"
                            B_bad = opb == identity ? rand(elty, m+1, n) : rand(elty, n, m+1)
                            error_str = opb == identity ? "first dimensions of A ($m) and X ($(m+1)) must match when transxy is 'N'" : "first dimension of A ($m) must match second dimension of X ($(m+1)) when transxy is not 'N'"
                            @test_throws DimensionMismatch(error_str) ldiv!(triangle(opa(dA)), opb(CuArray(B_bad)))
                        end
                    end
                end
            end

            @testset "$triangle($SparseMatrixType) \\ CuMatrix -- $elty" for elty in [Float64,ComplexF64]
                @testset "opa = $opa" for opa in [identity, transpose, adjoint]
                    @testset "opb = $opb" for opb in [identity, transpose, adjoint]
                        SparseMatrixType == CuSparseMatrixCSC && elty <: Complex && opa == adjoint && continue
                        elty <: Complex && opb == adjoint && continue
                        m = 10
                        n = 2
                        A = rand(elty, m, m)
                        A = triangle in (UnitLowerTriangular, LowerTriangular) ? tril(A) : triu(A)
                        A = triangle in (UnitLowerTriangular, UnitUpperTriangular) ? A - Diagonal(A) + I : A
                        A = sparse(A)
                        B = opb == identity ? rand(elty, m, n) : rand(elty, n, m)
                        dA = SparseMatrixType(A)
                        dB = CuArray(B)
                        C = triangle(opa(A)) \ opb(B)
                        dC = triangle(opa(dA)) \ opb(dB)
                        @test C ≈ collect(dC)
                        if CUSPARSE.version() < v"12.0"
                            B_bad = opb == identity ? rand(elty, m+1, n) : rand(elty, n, m+1)
                            error_str = opb == identity ? "first dimensions of A ($m) and X ($(m+1)) must match when transxy is 'N'" : "first dimension of A ($m) must match second dimension of X ($(m+1)) when transxy is not 'N'"
                            @test_throws DimensionMismatch(error_str) ldiv!(triangle(opa(dA)), opb(CuArray(B_bad)))
                        end
                    end
                end
            end
        end
    end
end

@testset "SparseArrays" begin
    @testset "spdiagm(CuVector{$elty})" for elty in [Float32, Float64, ComplexF32, ComplexF64]
        ref_vec = collect(elty, 100:121)
        cuda_vec = CuVector(ref_vec)

        ref_spdiagm = spdiagm(ref_vec) # SparseArrays
        cuda_spdiagm = spdiagm(cuda_vec)

        ref_cuda_sparse = CuSparseMatrixCSC(ref_spdiagm)

        @test ref_cuda_sparse.rowVal == cuda_spdiagm.rowVal

        @test ref_cuda_sparse.nzVal == cuda_spdiagm.nzVal

        @test ref_cuda_sparse.colPtr == cuda_spdiagm.colPtr
    end

    @testset "spdiagm(2 => CuVector{$elty})" for elty in [Float32, Float64, ComplexF32, ComplexF64]
        ref_vec = collect(elty, 100:121)
        cuda_vec = CuVector(ref_vec)

        ref_spdiagm = spdiagm(2 => ref_vec) # SparseArrays
        cuda_spdiagm = spdiagm(2 => cuda_vec)

        ref_cuda_sparse = CuSparseMatrixCSC(ref_spdiagm)

        @test ref_cuda_sparse.rowVal == cuda_spdiagm.rowVal

        @test ref_cuda_sparse.nzVal == cuda_spdiagm.nzVal

        @test ref_cuda_sparse.colPtr == cuda_spdiagm.colPtr
    end
end
