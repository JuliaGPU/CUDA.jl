using CUDA.CUSPARSE

@testset "similar" begin
    @testset "$f(A)±$h(B) $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64],
                                     f in (identity, transpose), #adjoint),
                                     (h,dimPtr,dimVal) in ((CuSparseMatrixCSR,:rowPtr,:colVal), (CuSparseMatrixCSC,:colPtr,:rowVal))
        n = 10
        A = sprand(elty, n, n, rand())
        dA = f(A)
        
        C_simple = similar(dA)
        C_dims = similar(dA,(n,n+1))
        C_eltype = similar(dA,elty)
        C_full = similar(dA,elty,(n,n+1))
        
    @test typeof(C_simple) ≈ typeof(dA)
    @test getproperty(C_simple,dimPtr) ≈ getproperty(dA,dimPtr) #similar without arguments perserves structure
    @test getproperty(C_simple,dimVal) ≈ getproperty(dA,dimVal)
    @test getproperty(C_simple,dims) ≈ getproperty(dA,dims)
    
    @test typeof(C_simple) ≈ typeof(dA)
    @test getproperty(C_simple,dimPtr) ≈ getproperty(dA,dimPtr) #similar without arguments perserves structure
    @test getproperty(C_simple,dimVal) ≈ getproperty(dA,dimVal)
    @test getproperty(C_dims,dims) ≈ (n,n+1)
    for propertyname in propertynames(dA)
      @test length(getproperty(C_simple,z)) ≈ length(getproperty(dA,z))
    end
        @test C ≈ collect(dC)

        C = f(A) - h(B)
        dC = f(dA) - h(dB)
        @test C ≈ collect(dC)
    end

    @testset "A±$f(B) $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64],
                                 f in (CuSparseMatrixCSR, CuSparseMatrixCSC, x->CuSparseMatrixBSR(x,1))
        n = 10
        A = sprand(elty, n, n, rand())
        B = sprand(elty, n, n, rand())

        dA = CuSparseMatrixCSR(A)
        dB = CuSparseMatrixCSR(B)

        C = A + B
        dC = dA + f(dB)
        @test C ≈ collect(dC)

        C = B + A
        dC = f(dB) + dA
        @test C ≈ collect(dC)

        C = A - B
        dC = dA - f(dB)
        @test C ≈ collect(dC)

        C = B - A
        dC = f(dB) - dA
        @test C ≈ collect(dC)
    end

    @testset "dense(A)$(op)sparse(B) $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64],
                                                op in [+, -]
        n = 10
        A = rand(elty, n, n)
        B = sprand(elty, n, n, rand())

        dA = CuArray(A)
        dB = CuSparseMatrixCSR(B)

        C = op(A, B)
        dC = op(dA, dB)
        @test C ≈ collect(dC)
        @test dC isa CuMatrix{elty}
        
        C = op(B, A)
        dC = op(dB, dA)
        @test C ≈ collect(dC)
        @test dC isa CuMatrix{elty}
    end

    @testset "$f(A)*b $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64],
                                 f in (identity, transpose, adjoint)
        n = 10
        alpha = rand()
        beta = rand()
        A = sprand(elty, n, n, rand())
        b = rand(elty, n)
        c = rand(elty, n)
        alpha = beta = 1.0
        c = zeros(elty, n)

        dA = CuSparseMatrixCSR(A)
        db = CuArray(b)
        dc = CuArray(c)

        # test with empty inputs
        @test Array(dA * CUDA.zeros(elty, n, 0)) == zeros(elty, n, 0)

        mul!(c, f(A), b, alpha, beta)
        mul!(dc, f(dA), db, alpha, beta)
        @test c ≈ collect(dc)

        A = A + transpose(A)
        dA = CuSparseMatrixCSR(A)

        @assert issymmetric(A)
        mul!(c, f(Symmetric(A)), b, alpha, beta)
        mul!(dc, f(Symmetric(dA)), db, alpha, beta)
        @test c ≈ collect(dc)
    end

    @testset "$f(A)*$h(B) $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64],
                                     f in (identity, transpose, adjoint),
                                     h in (identity, transpose, adjoint)
        if CUSPARSE.version() <= v"10.3.1" &&
            (h ∈ (transpose, adjoint) && f != identity) ||
            (h == adjoint && elty <: Complex)
            # These combinations are not implemented in CUDA10
            continue
        end

        n = 10
        alpha = rand()
        beta = rand()
        A = sprand(elty, n, n, rand())
        B = rand(elty, n, n)
        C = rand(elty, n, n)

        dA = CuSparseMatrixCSR(A)
        dB = CuArray(B)
        dC = CuArray(C)

        mul!(C, f(A), h(B), alpha, beta)
        mul!(dC, f(dA), h(dB), alpha, beta)
        @test C ≈ collect(dC)

        A = A + transpose(A)
        dA = CuSparseMatrixCSR(A)

        @assert issymmetric(A)
        mul!(C, f(Symmetric(A)), h(B), alpha, beta)
        mul!(dC, f(Symmetric(dA)), h(dB), alpha, beta)
        @test C ≈ collect(dC)
    end

    @testset "dense(A)*sparse(B) $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64]
        n = 10
        A = rand(elty, n, n)
        B = sprand(elty, n, n, rand())
    
        dA = CuArray(A)
        dB = CUSPARSE.CuSparseMatrixCSR(B)
    
        C = A * B
        dC = dA * dB
        @test C ≈ collect(dC)
        @test dC isa CuMatrix{elty}
        
        C = B * A
        dC = dB * dA
        @test C ≈ collect(dC)
        @test dC isa CuMatrix{elty}
    
        C = B * B
        dC = dB * dB
        @test C ≈ collect(dC)
        @test dC isa CuMatrix{elty}
    end

    @testset "issue #1095 ($elty)" for elty in [Float32, Float64, ComplexF32, ComplexF64]
        # Test non-square matrices
        n, m, p = 10, 20, 4
        alpha = rand()
        beta = rand()
        A = sprand(elty, n, m, rand())
        B = rand(elty, m, p)
        C = rand(elty, n, p)

        dA = CuSparseMatrixCSR(A)
        dB = CuArray(B)
        dC = CuArray(C)

        mul!(C, A, B, alpha, beta)
        mul!(dC, dA, dB, alpha, beta)
        @test C ≈ collect(dC)

        mul!(B, transpose(A), C, alpha, beta)
        mul!(dB, transpose(dA), dC, alpha, beta)
        @test B ≈ collect(dB)

        mul!(B, adjoint(A), C, alpha, beta)
        mul!(dB, adjoint(dA), dC, alpha, beta)
        @test B ≈ collect(dB)
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
end
