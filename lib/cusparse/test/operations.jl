@testset "mv!" begin
    @testset for elty in [Float32,Float64,ComplexF32,ComplexF64]
        A = sparse(rand(elty,m,n))
        x = rand(elty,n)
        y = rand(elty,m)
        alpha = rand(elty)
        beta = rand(elty)
        @testset "$(typeof(d_A))" for d_A in [CuSparseMatrixCSR(A),
                                              CuSparseMatrixCSC(A),
                                              CuSparseMatrixBSR(A, blockdim)]
            d_x = CuArray(x)
            d_y = CuArray(y)
            @test_throws DimensionMismatch cuSPARSE.mv!('T',alpha,d_A,d_x,beta,d_y,'O')
            @test_throws DimensionMismatch cuSPARSE.mv!('N',alpha,d_A,d_y,beta,d_x,'O')
            cuSPARSE.mv!('N',alpha,d_A,d_x,beta,d_y,'O')
            h_z = collect(d_y)
            z = alpha * A * x + beta * y
            @test z ≈ h_z
            #if d_A isa CuSparseMatrixCSR
            #    @test d_y' * (d_A * d_x) ≈ (d_y' * d_A) * d_x
            #end
        end
    end

    if capability(device()) >= v"5.3"
    @testset for elty in [Float16,ComplexF16]
        A = sparse(rand(elty,m,n))
        x = rand(elty,n)
        y = rand(elty,m)
        alpha = rand(elty)
        beta = rand(elty)
        @testset "$(typeof(d_A))" for d_A in [CuSparseMatrixCSR(A),
                                              CuSparseMatrixCSC(A)]
            d_x = CuArray(x)
            d_y = CuArray(y)
            @test_throws DimensionMismatch cuSPARSE.mv!('T',alpha,d_A,d_x,beta,d_y,'O')
            @test_throws DimensionMismatch cuSPARSE.mv!('N',alpha,d_A,d_y,beta,d_x,'O')
            cuSPARSE.mv!('N',alpha,d_A,d_x,beta,d_y,'O')
            h_z = collect(d_y)
            z = alpha * A * x + beta * y
            @test z ≈ h_z
            if d_A isa CuSparseMatrixCSR
                @test d_y' * (d_A * d_x) ≈ (d_y' * d_A) * d_x
            end
        end
    end
    end
end

@testset "mm!" begin
    @testset for elty in [Float32,Float64,ComplexF32,ComplexF64]
        A = sparse(rand(elty,m,k))
        B = rand(elty,k,n)
        C = rand(elty,m,n)
        alpha = rand(elty)
        beta = rand(elty)
        @testset "$(typeof(d_A))" for d_A in [CuSparseMatrixCSR(A),
                                              CuSparseMatrixCSC(A)]
            d_B = CuArray(B)
            d_C = CuArray(C)
            @test_throws DimensionMismatch cuSPARSE.mm!('N','T',alpha,d_A,d_B,beta,d_C,'O')
            @test_throws DimensionMismatch cuSPARSE.mm!('T','N',alpha,d_A,d_B,beta,d_C,'O')
            @test_throws DimensionMismatch cuSPARSE.mm!('T','T',alpha,d_A,d_B,beta,d_C,'O')
            @test_throws DimensionMismatch cuSPARSE.mm!('N','N',alpha,d_A,d_B,beta,d_B,'O')
            cuSPARSE.mm!('N','N',alpha,d_A,d_B,beta,d_C,'O')
            h_D = collect(d_C)
            D = alpha * A * B + beta * C
            @test D ≈ h_D
        end
    end

    @testset "issue 493" begin
        x = cu(rand(20))
        cu(sprand(Float32,10,10,0.1)) * @view(x[1:10])
    end
end

@testset "gemvi!" begin
    @testset for elty in [Float32,Float64,ComplexF32,ComplexF64]
        for (transa, opa) in [('N', identity), ('T', transpose), ('C', adjoint)]
            elty <: Complex && transa == 'C' && continue
            A = transa == 'N' ? rand(elty,m,n) : rand(elty,n,m)
            x = sparsevec(rand(1:n,k), rand(elty,k), n)
            y = rand(elty,m)
            dA = CuArray(A)
            dx = CuSparseVector(x)
            dy = CuArray(y)
            alpha = rand(elty)
            beta = rand(elty)
            gemvi!(transa,alpha,dA,dx,beta,dy,'O')
            z = alpha * opa(A) * x + beta * y
            @test z ≈ collect(dy)
        end
    end
end

for SparseMatrixType in [CuSparseMatrixCSC, CuSparseMatrixCSR]
    @testset "$SparseMatrixType -- color" begin
        @testset "color $T" for T in [Float32, Float64, ComplexF32, ComplexF64]
            A = sprand(T, 50, 50, 0.03)
            A = A + A'
            dA = SparseMatrixType(A)
            ncolors, coloring, reordering = color(dA, 'O')
            @test 1 ≤ ncolors ≤ 50
            @test maximum(coloring) == ncolors
            @test minimum(reordering) == 1
            @test maximum(reordering) == 50
            @test CUDACore.@allowscalar isperm(reordering)
        end

        # The routine color returns the wrong result with small sparse matrices.
        # NVIDIA investigates the issue.
        if false
            A = [ 1 1 0 0 0 ;
                  1 1 1 1 0 ;
                  0 1 1 0 1 ;
                  0 1 0 1 0 ;
                  0 0 1 0 1 ]
            A = sparse(A)
            # The adjacency graph of A has at least two colors, one color for
            # {1, 3, 4} and another one for {2, 5}.
            @testset "5x5 example -- color $T" for T in [Float32, Float64, ComplexF32, ComplexF64]
                dA = SparseMatrixType{T}(A)
                ncolors, coloring, reordering = color(dA, 'O')
                @test ncolors == 2
                @test minimum(reordering) == 1
                @test maximum(reordering) == 5
                CUDACore.allowscalar() do
                    @test isperm(reordering)
                    @test coloring[1] == coloring[3]  == coloring[4]
                    @test coloring[2] == coloring[5]
                    @test coloring[1] != coloring[2]
                end
            end
        end
    end
end
