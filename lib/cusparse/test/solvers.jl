@testset "bsrsv2" begin
    @testset for elty in [Float32,Float64,ComplexF32,ComplexF64]
        @testset "bsrsv2!" begin
            for unit_diag ∈ (false, true)
                diag = unit_diag ? 'U' : 'N'
                A = rand(elty,m,m)
                A = (unit_diag ? triu(A, 1) + I : triu(A))
                X = rand(elty,m)
                alpha = rand(elty)
                d_X = CuArray(X)
                d_A = CuSparseMatrixCSR(sparse(A))
                d_A = CuSparseMatrixBSR(d_A, blockdim)
                d_X = cuSPARSE.sv2!('N','U',diag,alpha,d_A,d_X,'O')
                h_Y = collect(d_X)
                Y = A\(alpha * X)
                @test Y ≈ h_Y
                d_X = CuArray(rand(elty,n))
                @test_throws DimensionMismatch cuSPARSE.sv2!('N','U',diag,alpha,d_A,d_X,'O')
                A = sparse(rand(elty,m,n))
                d_A = CuSparseMatrixCSR(A)
                d_A = CuSparseMatrixBSR(d_A, blockdim)
                @test_throws DimensionMismatch cuSPARSE.sv2!('N','U',diag,alpha,d_A,d_X,'O')
            end
        end

        @testset "bsrsv2" begin
            for unit_diag ∈ (false, true)
                diag = unit_diag ? 'U' : 'N'
                A = rand(elty,m,m)
                A = (unit_diag ? triu(A, 1) + I : triu(A))
                Al = (unit_diag ? tril(A, -1) + I : tril(A))
                X = rand(elty,m)
                alpha = rand(elty)
                d_X = CuArray(X)
                d_Al = CuSparseMatrixCSR(sparse(Al))
                d_Al = CuSparseMatrixBSR(d_Al, blockdim)
                d_Y = cuSPARSE.sv2('N','L',diag,alpha,d_Al,d_X,'O')
                h_Y = collect(d_Y)
                Y = Al\(alpha * X)
                @test Y ≈ h_Y
                d_A = CuSparseMatrixCSR(sparse(A))
                d_A = CuSparseMatrixBSR(d_A, blockdim)
                d_Y = cuSPARSE.sv2('N','U',diag,alpha,d_A,d_X,'O')
                h_Y = collect(d_Y)
                Y = A\(alpha * X)
                @test Y ≈ h_Y
                UA = (unit_diag ? UnitUpperTriangular(d_A) : UpperTriangular(d_A))
                d_Y = UA\d_X
                h_Y = collect(d_Y)
                @test h_Y ≈ A\X
                #=d_Y = UpperTriangular(d_A)'\d_X
                h_Y = collect(d_Y)
                @test h_Y ≈ A'\X=#
                d_Y = transpose(UA)\d_X
                h_Y = collect(d_Y)
                @test h_Y ≈ transpose(A)\X
                LA = (unit_diag ? UnitLowerTriangular(d_A) : LowerTriangular(d_A))
                d_Y = LA\d_X
                h_Y = collect(d_Y)
                @test h_Y ≈ Al\X
                #=d_Y = LowerTriangular(d_A)'\d_X
                h_Y = collect(d_Y)
                @test h_Y ≈ A'\X=#
                d_Y = transpose(LA)\d_X
                h_Y = collect(d_Y)
                @test h_Y ≈ transpose(Al)\X
                A = sparse(rand(elty,m,n))
                d_A = CuSparseMatrixCSR(A)
                d_A = CuSparseMatrixBSR(d_A, blockdim)
                @test_throws DimensionMismatch cuSPARSE.sv2('N','U',diag,alpha,d_A,d_X,'O')
            end
        end
    end
end

@testset "bsrsm2" begin
    @testset for elty in [Float32,Float64,ComplexF32,ComplexF64]
        @testset "bsrsm2!" begin
            for unit_diag ∈ (false, true)
                diag = unit_diag ? 'U' : 'N'
                A = rand(elty,m,m)
                A = (unit_diag ? triu(A, 1) + I : triu(A))
                X = rand(elty,m,m)
                alpha = rand(elty)
                d_X = CuArray(X)
                d_A = CuSparseMatrixCSR(sparse(A))
                d_A = CuSparseMatrixBSR(d_A, blockdim)
                d_X = cuSPARSE.sm2!('N','N','U',diag,alpha,d_A,d_X,'O')
                h_Y = collect(d_X)
                Y = A\(alpha * X)
                @test Y ≈ h_Y
                d_X = CuArray(rand(elty,n,n))
                @test_throws DimensionMismatch cuSPARSE.sm2!('N','N','U',diag,alpha,d_A,d_X,'O')
                A = sparse(rand(elty,m,n))
                d_A = CuSparseMatrixCSR(A)
                d_A = CuSparseMatrixBSR(d_A, blockdim)
                @test_throws DimensionMismatch cuSPARSE.sm2!('N','N','U',diag,alpha,d_A,d_X,'O')
            end
        end

        @testset "bsrsm2" begin
            for unit_diag ∈ (false, true)
                diag = unit_diag ? 'U' : 'N'
                A = rand(elty,m,m)
                A = (unit_diag ? triu(A, 1) + I : triu(A))
                Al = (unit_diag ? tril(A, -1) + I : tril(A))
                X = rand(elty,m, m)
                alpha = rand(elty)
                d_X = CuArray(X)
                d_Al = CuSparseMatrixCSR(sparse(Al))
                d_Al = CuSparseMatrixBSR(d_Al, blockdim)
                d_Y = cuSPARSE.sm2('N','N','L',diag,alpha,d_Al,d_X,'O')
                h_Y = collect(d_Y)
                Y = Al\(alpha * X)
                @test Y ≈ h_Y
                d_A = CuSparseMatrixCSR(sparse(A))
                d_A = CuSparseMatrixBSR(d_A, blockdim)
                d_Y = cuSPARSE.sm2('N','N','U',diag,alpha,d_A,d_X,'O')
                h_Y = collect(d_Y)
                Y = A\(alpha * X)
                @test Y ≈ h_Y
                UA = (unit_diag ? UnitUpperTriangular(d_A) : UpperTriangular(d_A))
                d_Y = UA\d_X
                h_Y = collect(d_Y)
                @test h_Y ≈ A\X
                #=d_Y = UpperTriangular(d_A)'\d_X
                h_Y = collect(d_Y)
                @test h_Y ≈ A'\X=#
                d_Y = transpose(UA)\d_X
                h_Y = collect(d_Y)
                @test h_Y ≈ transpose(A)\X
                LA = (unit_diag ? UnitLowerTriangular(d_A) : LowerTriangular(d_A))
                d_Y = LA\d_X
                h_Y = collect(d_Y)
                @test h_Y ≈ Al\X
                #=d_Y = LowerTriangular(d_A)'\d_X
                h_Y = collect(d_Y)
                @test h_Y ≈ A'\X=#
                d_Y = transpose(LA)\d_X
                h_Y = collect(d_Y)
                @test h_Y ≈ transpose(Al)\X
                A = sparse(rand(elty,m,n))
                d_A = CuSparseMatrixCSR(A)
                d_A = CuSparseMatrixBSR(d_A, blockdim)
                @test_throws DimensionMismatch cuSPARSE.sm2('N','N','U',diag,alpha,d_A,d_X,'O')
            end
        end
    end
end

@testset "Triangular solves -- CuSparseMatrixBSR" begin
    @testset "y = T \\ x -- $elty" for elty in (Float32, Float64, ComplexF32, ComplexF64)
        for (trans, op) in (('N', identity), ('T', transpose), ('C', adjoint))
            for uplo in ('L', 'U')
                for diag in ('N', 'U')
                    @testset "trans = $trans | uplo = $uplo | diag = $diag" begin
                        T = rand(elty,n,n)
                        T = uplo == 'L' ? tril(T) : triu(T)
                        T = diag == 'N' ? T : T - Diagonal(T) + I
                        T = sparse(T)
                        d_T = CuSparseMatrixBSR(CuSparseMatrixCSR(T), blockdim)
                        x = rand(elty,n)
                        d_x = CuVector{elty}(x)
                        d_y = cuSPARSE.sv2(trans, uplo, diag, d_T, d_x, 'O')
                        y = op(T) \ x
                        @test collect(d_y) ≈ y
                    end
                end
            end
        end

        @testset "Y = T \\ X -- $elty" for elty in (Float32, Float64, ComplexF32, ComplexF64)
            for (transT, opT) in (('N', identity), ('T', transpose), ('C', adjoint))
                for (transX, opX) in (('N', identity), ('T', transpose))
                    for uplo in ('L', 'U')
                        for diag in ('N', 'U')
                            @testset "transT = $transT | transX = $transX | uplo = $uplo | diag = $diag" begin
                                T = rand(elty,n,n)
                                T = uplo == 'L' ? tril(T) : triu(T)
                                T = diag == 'N' ? T : T - Diagonal(T) + I
                                T = sparse(T)
                                d_T = CuSparseMatrixBSR(CuSparseMatrixCSR(T), blockdim)
                                X = transX == 'N' ? rand(elty,n,p) : rand(elty,p,n)
                                d_X = CuMatrix{elty}(X)
                                d_Y = cuSPARSE.sm2(transT, transX, uplo, diag, d_T, d_X, 'O')
                                Y = opT(T) \ opX(X)
                                @test collect(d_Y) ≈ (transX == 'N' ? Y : transpose(Y))
                            end
                        end
                    end
                end
            end
        end
    end
end
