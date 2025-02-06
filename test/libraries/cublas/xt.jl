using CUDA.CUBLAS
using CUDA.CUBLAS: band, bandex

using LinearAlgebra

using BFloat16s
using StaticArrays

@test CUBLAS.version() isa VersionNumber
@test CUBLAS.version().major == CUBLAS.cublasGetProperty(CUDA.MAJOR_VERSION)
@test CUBLAS.version().minor == CUBLAS.cublasGetProperty(CUDA.MINOR_VERSION)
@test CUBLAS.version().patch == CUBLAS.cublasGetProperty(CUDA.PATCH_LEVEL)

m = 20
n = 35
k = 13

@testset "cublasXt" begin
    @testset for elty in [Float32, Float64, ComplexF32, ComplexF64]
        @testset "xt_trmm! gpu" begin
            alpha = rand(elty)
            A = triu(rand(elty, m, m))
            B = rand(elty,m,n)
            C = zeros(elty,m,n)
            dA = CuArray(A)
            dB = CuArray(B)
            dC = CuArray(C)
            C = alpha*A*B
            CUBLAS.xt_trmm!('L','U','N','N',alpha,dA,dB,dC)
            # move to host and compare
            h_C = Array(dC)
            @test C ≈ h_C
        end
        @testset "xt_trmm! cpu" begin
            alpha = rand(elty)
            A = triu(rand(elty, m, m))
            B = rand(elty,m,n)
            C = alpha*A*B
            h_C = zeros(elty, m, n)
            CUBLAS.xt_trmm!('L','U','N','N',alpha,copy(A),copy(B),h_C)
            @test C ≈ h_C
        end
        @testset "xt_trmm gpu" begin
            alpha = rand(elty)
            A = triu(rand(elty, m, m))
            B = rand(elty,m,n)
            C = zeros(elty,m,n)
            dA = CuArray(A)
            dB = CuArray(B)
            dC = CuArray(C)
            C = alpha*A*B
            d_C = CUBLAS.xt_trmm('L','U','N','N',alpha,dA,dB)
            # move to host and compare
            @test d_C isa CuArray
            h_C = Array(d_C)
            @test C ≈ h_C
        end
        @testset "xt_trmm cpu" begin
            alpha = rand(elty)
            A = triu(rand(elty, m, m))
            B = rand(elty,m,n)
            C = alpha*A*B
            h_C = CUBLAS.xt_trmm('L','U','N','N',alpha,copy(A),copy(B))
            @test h_C isa Array
            @test C ≈ h_C
        end

        @testset "xt_trsm! gpu" begin
            alpha = rand(elty)
            A = triu(rand(elty, m, m))
            B = rand(elty,m,n)
            dA = CuArray(A)
            dB = CuArray(B)
            C = alpha*(A\B)
            dC = copy(dB)
            synchronize()
            CUBLAS.xt_trsm!('L','U','N','N',alpha,dA,dC)
            # move to host and compare
            h_C = Array(dC)
            @test C ≈ h_C
        end
        @testset "xt_symm! gpu" begin
            alpha = rand(elty)
            beta = rand(elty)
            sA = rand(elty,m,m)
            sA = sA + transpose(sA)
            dsA = CuArray(sA)
            B = rand(elty,m,n)
            C = rand(elty,m,n)
            Bbad = rand(elty,m+1,n+1)
            d_B = CuArray(B)
            d_C = CuArray(C)
            CUBLAS.xt_symm!('L','U',alpha,dsA,d_B,beta,d_C)
            C = (alpha*sA)*B + beta*C
            # compare
            h_C = Array(d_C)
            @test C ≈ h_C
        end
        @testset "xt_symm! cpu" begin
            alpha = rand(elty)
            beta = rand(elty)
            sA = rand(elty,m,m)
            sA = sA + transpose(sA)
            B = rand(elty,m,n)
            C = rand(elty,m,n)
            h_C = copy(C) 
            CUBLAS.xt_symm!('L','U',alpha,copy(sA),copy(B),beta,h_C)
            C = (alpha*sA)*B + beta*C
            # compare
            @test C ≈ h_C
        end

        @testset "xt_symm gpu" begin
            sA = rand(elty,m,m)
            sA = sA + transpose(sA)
            dsA = CuArray(sA)
            B = rand(elty,m,n)
            d_B = CuArray(B)
            d_C = CUBLAS.xt_symm('L','U',dsA,d_B)
            C = sA*B
            # compare
            @test d_C isa CuArray
            h_C = Array(d_C)
            @test C ≈ h_C
        end
        @testset "xt_symm cpu" begin
            sA = rand(elty,m,m)
            sA = sA + transpose(sA)
            B = rand(elty,m,n)
            h_C = CUBLAS.xt_symm('L','U',copy(sA),copy(B))
            C = sA*B
            # compare
            @test h_C isa Array
            @test C ≈ h_C
        end
        @testset "xt_gemm! gpu" begin
            alpha = rand(elty)
            beta = rand(elty)
            A = rand(elty,m,k)
            B = rand(elty,k,n)
            C1  = rand(elty,m,n)
            C2  = copy(C1) 
            d_A = CuArray(A)
            d_B = CuArray(B)
            Bbad = rand(elty,k+1,n+1)
            d_Bbad = CuArray(Bbad)
            d_C1 = CuArray(C1)
            d_C2 = CuArray(C2)
            @test_throws DimensionMismatch CUBLAS.xt_gemm!('N','N',alpha,d_A,d_Bbad,beta,d_C1)
            synchronize()
            CUBLAS.xt_gemm!('N','N',alpha,d_A,d_B,beta,d_C1)
            mul!(d_C2, d_A, d_B)
            h_C1 = Array(d_C1)
            h_C2 = Array(d_C2)
            C1 = (alpha*A)*B + beta*C1
            C2 = A*B
            # compare
            @test C1 ≈ h_C1
            @test C2 ≈ h_C2
        end
        @testset "xt_gemm! cpu" begin
            alpha = rand(elty)
            beta  = rand(elty)
            A     = rand(elty,m,k)
            B     = rand(elty,k,n)
            C1    = rand(elty,m,n)
            C2    = copy(C1)
            C3    = copy(C1)
            C4    = copy(C2)
            CUBLAS.xt_gemm!('N','N',alpha,A,B,beta,C1)
            mul!(C2, A, B)
            C3 = (alpha*A)*B + beta*C3
            C4 = A*B
            # compare
            @test C1 ≈ C3
            @test C2 ≈ C4
        end
        @testset "xt_gemm gpu" begin
            A = rand(elty,m,k)
            B = rand(elty,k,n)
            d_A = CuArray(A)
            d_B = CuArray(B)
            synchronize()
            d_C = CUBLAS.xt_gemm('N','N',d_A,d_B)
            C  = A*B
            C2 = d_A * d_B
            # compare
            @test d_C isa CuArray
            h_C = Array(d_C)
            h_C2 = Array(C2)
            @test C ≈ h_C
            @test C ≈ h_C2
        end
        @testset "xt_gemm cpu" begin
            A = rand(elty,m,k)
            B = rand(elty,k,n)
            C = CUBLAS.xt_gemm('N','N',A,B)
            C2  = A*B
            # compare
            @test C isa Array
            @test C ≈ A*B
            @test C ≈ C2
        end
        @testset "xt_trsm! cpu" begin
            alpha = rand(elty)
            A = triu(rand(elty, m, m))
            B = rand(elty,m,n)
            C = alpha*(A\B)
            h_C = copy(B)
            synchronize()
            CUBLAS.xt_trsm!('L','U','N','N',alpha,copy(A),h_C)
            @test C ≈ h_C
        end
        @testset "xt_trsm gpu" begin
            alpha = rand(elty)
            A = triu(rand(elty, m, m))
            B = rand(elty,m,n)
            dA = CuArray(A)
            dB = CuArray(B)
            C  = alpha*(A\B)
            synchronize()
            dC = CUBLAS.xt_trsm('L','U','N','N',alpha,dA,dB)
            # move to host and compare
            @test dC isa CuArray
            h_C = Array(dC)
            @test C ≈ h_C
        end
        @testset "xt_trsm cpu" begin
            alpha = rand(elty)
            A = triu(rand(elty, m, m))
            B = rand(elty,m,n)
            C  = alpha*(A\B)
            synchronize()
            h_C = CUBLAS.xt_trsm('L','U','N','N',alpha,copy(A),copy(B))
            @test h_C isa Array
            @test C ≈ h_C
        end
        @testset "xt_syrkx! gpu" begin
            alpha = rand(elty)
            beta = rand(elty)
            # generate matrices
            syrkx_A = rand(elty, n, k)
            syrkx_B = rand(elty, n, k)
            syrkx_C = rand(elty, n, n)
            syrkx_C += syrkx_C'
            d_syrkx_A = CuArray(syrkx_A)
            d_syrkx_B = CuArray(syrkx_B)
            d_syrkx_C = CuArray(syrkx_C)
            # C = (alpha*A)*transpose(B) + beta*C
            synchronize()
            d_syrkx_C = CUBLAS.xt_syrkx!('U','N',alpha,d_syrkx_A,d_syrkx_B,beta,d_syrkx_C)
            final_C = (alpha*syrkx_A)*transpose(syrkx_B) + beta*syrkx_C
            # move to host and compare
            h_C = Array(d_syrkx_C)
            @test triu(final_C) ≈ triu(h_C)
            badC = rand(elty, m, n)
            d_badC = CuArray(badC)
            @test_throws DimensionMismatch CUBLAS.xt_syrkx!('U','N',alpha,d_syrkx_A,d_syrkx_B,beta,d_badC)
            badC = rand(elty, n+1, n+1)
            d_badC = CuArray(badC)
            @test_throws DimensionMismatch CUBLAS.xt_syrkx!('U','N',alpha,d_syrkx_A,d_syrkx_B,beta,d_badC)
        end
        @testset "xt_syrkx! cpu" begin
            alpha = rand(elty)
            beta = rand(elty)
            # generate matrices
            syrkx_A = rand(elty, n, k)
            syrkx_B = rand(elty, n, k)
            syrkx_C = rand(elty, n, n)
            syrkx_C += syrkx_C'
            final_C = (alpha*syrkx_A)*transpose(syrkx_B) + beta*syrkx_C
            CUBLAS.xt_syrkx!('U','N',alpha,syrkx_A,syrkx_B,beta,syrkx_C)
            # move to host and compare
            @test triu(final_C) ≈ triu(syrkx_C)
        end
        @testset "xt_syrkx gpu" begin
            # generate matrices
            syrkx_A = rand(elty, n, k)
            syrkx_B = rand(elty, n, k)
            d_syrkx_A = CuArray(syrkx_A)
            d_syrkx_B = CuArray(syrkx_B)
            synchronize()
            d_syrkx_C = CUBLAS.xt_syrkx('U','N',d_syrkx_A,d_syrkx_B)
            final_C = syrkx_A*transpose(syrkx_B)
            # move to host and compare
            @test d_syrkx_C isa CuArray
            h_C = Array(d_syrkx_C)
            @test triu(final_C) ≈ triu(h_C)
        end
        @testset "xt_syrkx cpu" begin
            # generate matrices
            syrkx_A = rand(elty, n, k)
            syrkx_B = rand(elty, n, k)
            h_C = CUBLAS.xt_syrkx('U','N',syrkx_A,syrkx_B)
            final_C = syrkx_A*transpose(syrkx_B)
            @test h_C isa Array
            @test triu(final_C) ≈ triu(h_C)
        end
        @testset "xt_syrk gpu" begin
            # C = A*transpose(A)
            A = rand(elty,m,k)
            d_A = CuArray(A)
            d_C = CUBLAS.xt_syrk('U','N',d_A)
            C = A*transpose(A)
            C = triu(C)
            # move to host and compare
            @test d_C isa CuArray
            h_C = Array(d_C)
            h_C = triu(C)
            @test C ≈ h_C
        end
        @testset "xt_syrk cpu" begin
            A = rand(elty,m,k)
            # C = A*transpose(A)
            h_C = CUBLAS.xt_syrk('U','N',copy(A))
            C = A*transpose(A)
            C = triu(C)
            # move to host and compare
            @test h_C isa Array
            h_C = triu(C)
            @test C ≈ h_C
        end
        if elty <: Complex
            @testset "xt_hemm! gpu" begin
                alpha = rand(elty)
                beta = rand(elty)
                hA = rand(elty,m,m)
                hA = hA + hA'
                dhA = CuArray(hA)
                B = rand(elty,m,n)
                C = rand(elty,m,n)
                d_B = CuArray(B)
                d_C = CuArray(C)
                # compute
                C = alpha*(hA*B) + beta*C
                CUBLAS.xt_hemm!('L','L',alpha,dhA,d_B,beta,d_C)
                # move to host and compare
                h_C = Array(d_C)
                @test C ≈ h_C
            end
            @testset "xt_hemm! cpu" begin
                alpha = rand(elty)
                beta = rand(elty)
                hA = rand(elty,m,m)
                hA = hA + hA'
                B = rand(elty,m,n)
                C = rand(elty,m,n)
                # compute
                h_C = copy(C)
                C = alpha*(hA*B) + beta*C
                CUBLAS.xt_hemm!('L','L',alpha,copy(hA),copy(B),beta,h_C)
                @test C ≈ h_C
            end
            @testset "xt_hemm gpu" begin
                hA  = rand(elty,m,m)
                hA  = hA + hA'
                dhA = CuArray(hA)
                B   = rand(elty,m,n)
                d_B = CuArray(B)
                C   = hA*B
                d_C = CUBLAS.xt_hemm('L','U',dhA, d_B)
                # move to host and compare
                @test d_C isa CuArray
                h_C = Array(d_C)
                @test C ≈ h_C
            end
            @testset "xt_hemm cpu" begin
                hA = rand(elty,m,m)
                hA = hA + hA'
                B = rand(elty,m,n)
                C   = hA*B
                h_C = CUBLAS.xt_hemm('L','U',copy(hA), copy(B))
                # move to host and compare
                @test h_C isa Array
                @test C ≈ h_C
            end
            @testset "xt_herk! gpu" begin
                alpha = rand(elty)
                beta = rand(elty)
                A = rand(elty,m,m)
                hA = A + A'
                C = real(alpha)*(A*A') + real(beta)*copy(hA)
                d_A = CuArray(A)
                d_C = CuArray(hA)
                synchronize()
                CUBLAS.xt_herk!('U','N',real(alpha),d_A,real(beta),d_C)
                C = triu(C)
                # move to host and compare
                h_C = Array(d_C)
                h_C = triu(h_C)
                @test C ≈ h_C
            end
            @testset "xt_herk! cpu" begin
                alpha = rand(elty)
                beta = rand(elty)
                A = rand(elty,m,m)
                hA = A + A'
                h_C = copy(hA)
                CUBLAS.xt_herk!('U','N',real(alpha),copy(A),real(beta),h_C)
                C = real(alpha)*(A*A') + real(beta)*copy(hA)
                C = triu(C)
                # move to host and compare
                h_C = triu(h_C)
                @test C ≈ h_C
            end
            @testset "xt_herk gpu" begin
                A = rand(elty,m,m)
                d_A = CuArray(A)
                synchronize()
                d_C = CUBLAS.xt_herk('U','N',d_A)
                C = A*A'
                C = triu(C)
                # move to host and compare
                @test d_C isa CuArray
                h_C = Array(d_C)
                h_C = triu(h_C)
                @test C ≈ h_C
            end
            @testset "xt_herk cpu" begin
                A = rand(elty,m,m)
                h_C = CUBLAS.xt_herk('U','N',copy(A))
                C = A*A'
                C = triu(C)
                # move to host and compare
                @test h_C isa Array
                h_C = triu(h_C)
                @test C ≈ h_C
            end
            @testset "xt_her2k! gpu" begin
                elty1 = elty
                elty2 = real(elty)
                # generate parameters
                α = rand(elty1)
                β = rand(elty2)
                A = rand(elty,m,k)
                B = rand(elty,m,k)
                d_A = CuArray(A)
                d_B = CuArray(B)
                C = rand(elty,m,m)
                C = C + C'
                d_C = CuArray(C)
                C = α*(A*B') + conj(α)*(B*A') + β*C
                synchronize()
                CUBLAS.xt_her2k!('U','N',α,d_A,d_B,β,d_C)
                # move back to host and compare
                C = triu(C)
                h_C = Array(d_C)
                h_C = triu(h_C)
                @test C ≈ h_C
            end
            @testset "xt_her2k! cpu" begin
                elty1 = elty
                elty2 = real(elty)
                # generate parameters
                α = rand(elty1)
                β = rand(elty2)
                A = rand(elty,m,k)
                B = rand(elty,m,k)
                C = rand(elty,m,m)
                C = C + C'
                h_C = copy(C)
                C = α*(A*B') + conj(α)*(B*A') + β*C
                CUBLAS.xt_her2k!('U','N',α,A,B,β,h_C)
                # move back to host and compare
                C = triu(C)
                h_C = triu(h_C)
                @test C ≈ h_C
            end
            @testset "xt_her2k gpu" begin
                # generate parameters
                A = rand(elty,m,k)
                B = rand(elty,m,k)
                d_A = CuArray(A)
                d_B = CuArray(B)
                C = rand(elty,m,m)
                C = C + C'
                C = (A*B') + (B*A')
                synchronize()
                d_C = CUBLAS.xt_her2k('U','N',d_A,d_B)
                # move back to host and compare
                C = triu(C)
                @test d_C isa CuArray
                h_C = Array(d_C)
                h_C = triu(h_C)
                @test C ≈ h_C
            end
            @testset "xt_her2k cpu" begin
                A = rand(elty,m,k)
                B = rand(elty,m,k)
                C = rand(elty,m,m)
                # generate parameters
                C = C + C'
                C = (A*B') + (B*A')
                h_C = CUBLAS.xt_her2k('U','N',A,B)
                # move back to host and compare
                @test h_C isa Array
                C = triu(C)
                h_C = triu(h_C)
                @test C ≈ h_C
            end
            @testset "her2k" begin
                A = rand(elty,m,k)
                B = rand(elty,m,k)
                d_A = CuArray(A)
                d_B = CuArray(B)
                C = A*B' + B*A'
                d_C = CUBLAS.her2k('U','N',d_A,d_B)
                # move back to host and compare
                C = triu(C)
                h_C = Array(d_C)
                h_C = triu(h_C)
                @test C ≈ h_C
            end
        end
    end
end
