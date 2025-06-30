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

@testset "level 3" begin
    @testset "conversion argument errors" begin
        @test_throws ArgumentError("Unknown operation D") convert(CUBLAS.cublasOperation_t, 'D')
        @test_throws ArgumentError("Unknown fill mode D") convert(CUBLAS.cublasFillMode_t, 'D')
        @test_throws ArgumentError("Unknown diag mode D") convert(CUBLAS.cublasDiagType_t, 'D')
        @test_throws ArgumentError("Unknown side mode D") convert(CUBLAS.cublasSideMode_t, 'D')
    end
    @testset for elty in [Float32, Float64, ComplexF32, ComplexF64]
        @testset "trmm!" begin
            alpha = rand(elty)
            A = triu(rand(elty, m, m))
            B = rand(elty,m,n)
            C = zeros(elty,m,n)
            dA = CuArray(A)
            dB = CuArray(B)
            dC = CuArray(C)
            C = alpha*A*B
            CUBLAS.trmm!('L','U','N','N',alpha,dA,dB,dC)
            # move to host and compare
            h_C = Array(dC)
            @test C ≈ h_C
        end
        @testset "trmm" begin
            alpha = rand(elty)
            A = triu(rand(elty, m, m))
            B = rand(elty,m,n)
            C = zeros(elty,m,n)
            dA = CuArray(A)
            dB = CuArray(B)
            C = alpha*A*B
            d_C = CUBLAS.trmm('L','U','N','N',alpha,dA,dB)
            # move to host and compare
            h_C = Array(d_C)
            @test C ≈ h_C
        end
        @testset "triangular-dense mul!" begin
            A = triu(rand(elty, m, m))
            B = rand(elty,m,n)
            C = zeros(elty,m,n)

            sA = rand(elty,m,m)
            sA = sA + transpose(sA)

            for t in (identity, transpose, adjoint), TR in (UpperTriangular, LowerTriangular, UnitUpperTriangular, UnitLowerTriangular)
                A = copy(sA) |> TR
                B_L = copy(B)
                B_R = copy(B')
                C_L = copy(C)
                C_R = copy(C')
                dA = CuArray(parent(A)) |> TR
                dB_L = CuArray(parent(B_L))
                dB_R = CuArray(parent(B_R))
                dC_L = CuArray(C_L)
                dC_R = CuArray(C_R)

                D_L = mul!(C_L, t(A), B_L)
                dD_L = mul!(dC_L, t(dA), dB_L)

                D_R = mul!(C_R, B_R, t(A))
                dD_R = mul!(dC_R, dB_R, t(dA))

                @test C_L ≈ Array(dC_L)
                @test D_L ≈ Array(dD_L)
                @test C_R ≈ Array(dC_R)
                @test D_R ≈ Array(dD_R)
            end
        end

        @testset "triangular-triangular mul!" begin
            A  = triu(rand(elty, m, m))
            B  = triu(rand(elty, m, m))
            C0 = zeros(elty,m,m)

            sA = rand(elty,m,m)
            sA = sA + transpose(sA)
            sB = rand(elty,m,m)
            sB = sB + transpose(sB)

            for (TRa, ta, TRb, tb, TRc, a_func, b_func) in (
                (UpperTriangular, identity,  LowerTriangular, identity,  Matrix, triu, tril),
                (LowerTriangular, identity,  UpperTriangular, identity,  Matrix, tril, triu),
                (UpperTriangular, identity,  UpperTriangular, transpose, Matrix, triu, triu),
                (UpperTriangular, transpose, UpperTriangular, identity,  Matrix, triu, triu),
                (LowerTriangular, identity,  LowerTriangular, transpose, Matrix, tril, tril),
                (LowerTriangular, transpose, LowerTriangular, identity,  Matrix, tril, tril),
                )

                A = copy(sA) |> TRa
                B = copy(sB) |> TRb
                C = copy(C0) |> TRc
                dA = CuArray(a_func(parent(sA))) |> TRa
                dB = CuArray(b_func(parent(sB))) |> TRb
                dC = if TRc == Matrix
                    CuArray(C0) |> DenseCuMatrix
                else
                    CuArray(C0) |> TRc
                end

                D = mul!(C, ta(A), tb(B))
                dD = mul!(dC, ta(dA), tb(dB))

                @test C ≈ Array(dC)
                @test D ≈ Array(dD)
            end
        end
        @testset "trsm" begin
            # compute
            @testset "adjtype=$adjtype, uplotype=$uplotype" for
                adjtype in (identity, adjoint, transpose),
                    uplotype in (UpperTriangular, UnitUpperTriangular, LowerTriangular, UnitLowerTriangular)

                A = triu(rand(elty, m, m))
                dA = CuArray(A)
                Br = rand(elty,m,n)
                Bl = rand(elty,n,m)
                d_Br = CuArray(Br)
                d_Bl = CuArray(Bl)
                @test adjtype(uplotype(A))\Br ≈ Array(adjtype(uplotype(dA))\d_Br)
                @test Bl/adjtype(uplotype(A)) ≈ Array(d_Bl/adjtype(uplotype(dA)))
            end
            # Check also that scaling parameter works
            alpha = rand(elty)
            A = triu(rand(elty, m, m))
            dA = CuArray(A)
            Br = rand(elty,m,n)
            d_Br = CuArray(Br)
            @test BLAS.trsm('L','U','N','N',alpha,A,Br) ≈ Array(CUBLAS.trsm('L','U','N','N',alpha,dA,d_Br))
        end

        @testset "trsm_batched!" begin
            alpha = rand(elty)
            bA = [rand(elty,m,m) for i in 1:10]
            map!((x) -> triu(x), bA, bA)
            bB = [rand(elty,m,n) for i in 1:10]
            bBbad = [rand(elty,m,n) for i in 1:9]
            # move to device
            bd_A = CuArray{elty, 2}[]
            bd_B = CuArray{elty, 2}[]
            bd_Bbad = CuArray{elty, 2}[]
            for i in 1:length(bA)
                push!(bd_A,CuArray(bA[i]))
                push!(bd_B,CuArray(bB[i]))
            end
            for i in 1:length(bBbad)
                push!(bd_Bbad,CuArray(bBbad[i]))
            end
            # compute
            CUBLAS.trsm_batched!('L','U','N','N',alpha,bd_A,bd_B)
            @test_throws DimensionMismatch CUBLAS.trsm_batched!('L','U','N','N',alpha,bd_A,bd_Bbad)
            # move to host and compare
            for i in 1:length(bd_B)
                bC = alpha*(bA[i]\bB[i])
                h_C = Array(bd_B[i])
                #compare
                @test bC ≈ h_C
            end
        end

        @testset "trsm_batched" begin
            # generate parameter
            alpha = rand(elty)
            # generate matrices
            bA = [rand(elty,m,m) for i in 1:10]
            map!((x) -> triu(x), bA, bA)
            bB = [rand(elty,m,n) for i in 1:10]
            # move to device
            bd_A = CuArray{elty, 2}[]
            bd_B = CuArray{elty, 2}[]
            for i in 1:length(bA)
                push!(bd_A,CuArray(bA[i]))
                push!(bd_B,CuArray(bB[i]))
            end
            # compute
            bd_C = CUBLAS.trsm_batched('L','U','N','N',alpha,bd_A,bd_B)
            # move to host and compare
            for i in 1:length(bd_C)
                bC = alpha*(bA[i]\bB[i])
                h_C = Array(bd_C[i])
                @test bC ≈ h_C
            end
        end

        let A = triu(rand(elty, m, m)), B = rand(elty,m,n), alpha = rand(elty)
            dA = CuArray(A)
            dB = CuArray(B)

            @testset "left trsm!" begin
                C = alpha*(A\B)
                dC = copy(dB)
                CUBLAS.trsm!('L','U','N','N',alpha,dA,dC)
                @test C ≈ Array(dC)
            end

            @testset "left trsm" begin
                C = alpha*(A\B)
                dC = CUBLAS.trsm('L','U','N','N',alpha,dA,dB)
                @test C ≈ Array(dC)
            end
            @testset "left trsm (adjoint)" begin
                C = alpha*(adjoint(A)\B)
                dC = CUBLAS.trsm('L','U','C','N',alpha,dA,dB)
                @test C ≈ Array(dC)
            end
            @testset "left trsm (transpose)" begin
                C = alpha*(transpose(A)\B)
                dC = CUBLAS.trsm('L','U','T','N',alpha,dA,dB)
                @test C ≈ Array(dC)
            end
        end

        @testset "triangular ldiv!" begin
            A = triu(rand(elty, m, m))
            B = rand(elty, m,m)

            dA = CuArray(A)
            dB = CuArray(B)

            for t in (identity, transpose, adjoint), TR in (UpperTriangular, LowerTriangular, UnitUpperTriangular, UnitLowerTriangular)
                dC = copy(dB)
                ldiv!(t(TR(dA)), dC)
                C = t(TR(A)) \ B
                @test C ≈ Array(dC)
            end
        end

        let A = rand(elty, m,m), B = triu(rand(elty, m, m)), alpha = rand(elty)
            dA = CuArray(A)
            dB = CuArray(B)

            @testset "right trsm!" begin
                C = alpha*(A/B)
                dC = copy(dA)
                CUBLAS.trsm!('R','U','N','N',alpha,dB,dC)
                @test C ≈ Array(dC)
            end

            @testset "right trsm" begin
                C = alpha*(A/B)
                dC = CUBLAS.trsm('R','U','N','N',alpha,dB,dA)
                @test C ≈ Array(dC)
            end
            @testset "right trsm (adjoint)" begin
                C = alpha*(A/adjoint(B))
                dC = CUBLAS.trsm('R','U','C','N',alpha,dB,dA)
                @test C ≈ Array(dC)
            end
            @testset "right trsm (transpose)" begin
                C = alpha*(A/transpose(B))
                dC = CUBLAS.trsm('R','U','T','N',alpha,dB,dA)
                @test C ≈ Array(dC)
            end
        end

        @testset "triangular rdiv!" begin
            A = rand(elty, m,m)
            B = triu(rand(elty, m, m))

            dA = CuArray(A)
            dB = CuArray(B)

            for t in (identity, transpose, adjoint), TR in (UpperTriangular, LowerTriangular, UnitUpperTriangular, UnitLowerTriangular)
                dC = copy(dA)
                rdiv!(dC, t(TR(dB)))
                C = A / t(TR(B))
                @test C ≈ Array(dC)
            end
        end

        @testset "Diagonal rdiv!" begin
            A = rand(elty, m,m)
            B = Diagonal(rand(elty, m))

            dA = CuArray(A)
            dB = adapt(CuArray, B)

            C = A / B
            d_C = dA / dB
            h_C = Array(d_C)
            rdiv!(dA, dB)
            @test C ≈ Array(dA)
            @test C ≈ h_C

            B_bad = Diagonal(CuArray(rand(elty, m+1)))
            @test_throws DimensionMismatch("left hand side has $m columns but D is $(m+1) by $(m+1)") rdiv!(dA, B_bad)
        end

        @testset "geam!" begin
            alpha = rand(elty)
            beta = rand(elty)
            A = rand(elty,m,n)
            B = rand(elty,m,n)
            C = zeros(elty,m,n)
            d_A = CuArray(A)
            d_B = CuArray(B)
            d_C = CuArray(C)
            # compute
            D = alpha*A + beta*B
            CUBLAS.geam!('N','N',alpha,d_A,beta,d_B,d_C)
            # move to host and compare
            h_C = Array(d_C)
            @test D ≈ h_C

            #test in place versions too
            d_C = CuArray(C)
            D = alpha*C + beta*B
            CUBLAS.geam!('N','N',alpha,d_C,beta,d_B,d_C)
            # move to host and compare
            h_C = Array(d_C)
            @test D ≈ h_C

            d_C = CuArray(C)
            D = alpha*A + beta*C
            CUBLAS.geam!('N','N',alpha,d_A,beta,d_C,d_C)
            # move to host and compare
            h_C = Array(d_C)
            @test D ≈ h_C

            #test setting C to zero
            CUBLAS.geam!('N','N',zero(elty),d_A,zero(elty),d_B,d_C)
            h_C = Array(d_C)
            @test h_C ≈ zeros(elty,m,n)

            # bounds checking
            @test_throws DimensionMismatch CUBLAS.geam!('N','T',alpha,d_A,beta,d_B,d_C)
            @test_throws DimensionMismatch CUBLAS.geam!('T','T',alpha,d_A,beta,d_B,d_C)
            @test_throws DimensionMismatch CUBLAS.geam!('T','N',alpha,d_A,beta,d_B,d_C)
        end

        @testset "geam" begin
            alpha = rand(elty)
            beta = rand(elty)
            A = rand(elty, m, n)
            d_A = CuArray(A)
            B = rand(elty, m, n)
            d_B = CuArray(B)
            D = alpha*A + beta*B
            d_C = CUBLAS.geam('N','N',alpha,d_A,beta,d_B)
            # move to host and compare
            h_C = Array(d_C)
            @test D ≈ h_C
            d_C = CUBLAS.geam('N','N',d_A,d_B)
            h_C = Array(d_C)
            D = A + B
            @test D ≈ h_C
        end
        @testset "CuMatrix -- A ± B -- $elty" begin
            for opa in (identity, transpose, adjoint)
                for opb in (identity, transpose, adjoint)
                    n = 10
                    m = 20
                    geam_A = opa == identity ? rand(elty, n, m) : rand(elty, m, n)
                    geam_B = opb == identity ? rand(elty, n, m) : rand(elty, m, n)

                    geam_dA = CuMatrix{elty}(geam_A)
                    geam_dB = CuMatrix{elty}(geam_B)

                    geam_C = opa(geam_A) + opb(geam_B)
                    geam_dC = opa(geam_dA) + opb(geam_dB)
                    @test geam_C ≈ collect(geam_dC)

                    geam_C = opa(geam_A) - opb(geam_B)
                    geam_dC = opa(geam_dA) - opb(geam_dB)
                    @test geam_C ≈ collect(geam_dC)
                end
            end
        end
        @testset "syrkx!" begin
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
            d_syrkx_C = CUBLAS.syrkx!('U','N',alpha,d_syrkx_A,d_syrkx_B,beta,d_syrkx_C)
            final_C = (alpha*syrkx_A)*transpose(syrkx_B) + beta*syrkx_C
            # move to host and compare
            h_C = Array(d_syrkx_C)
            @test triu(final_C) ≈ triu(h_C)
            # C = A*transpose(B)
            d_syrkx_C = CUBLAS.syrkx('U','N',d_syrkx_A,d_syrkx_B)
            final_C   = syrkx_A*transpose(syrkx_B)
            # move to host and compare
            h_C = Array(d_syrkx_C)
            @test triu(final_C) ≈ triu(h_C)
            badC = rand(elty, m, n)
            d_badC = CuArray(badC)
            @test_throws DimensionMismatch CUBLAS.syrkx!('U','N',alpha,d_syrkx_A,d_syrkx_B,beta,d_badC)
            badC = rand(elty, n+1, n+1)
            d_badC = CuArray(badC)
            @test_throws DimensionMismatch CUBLAS.syrkx!('U','N',alpha,d_syrkx_A,d_syrkx_B,beta,d_badC)
        end
        @testset "syrk" begin
            A = rand(elty,m,k)
            d_A = CuArray(A)
            # C = A*transpose(A)
            d_C = CUBLAS.syrk('U','N',d_A)
            C = A*transpose(A)
            C = triu(C)
            # move to host and compare
            h_C = Array(d_C)
            h_C = triu(C)
            @test C ≈ h_C
        end
        @testset "syr2k!" begin
            alpha = rand(elty)
            beta = rand(elty)
            A = rand(elty,m,k)
            B = rand(elty,m,k)
            Bbad = rand(elty,m+1,k+1)
            C = rand(elty,m,m)
            C = C + transpose(C)
            # move to device
            d_A = CuArray(A)
            d_B = CuArray(B)
            d_Bbad = CuArray(Bbad)
            d_C = CuArray(C)
            # compute
            C = alpha*(A*transpose(B) + B*transpose(A)) + beta*C
            CUBLAS.syr2k!('U','N',alpha,d_A,d_B,beta,d_C)
            # move back to host and compare
            C = triu(C)
            h_C = Array(d_C)
            h_C = triu(h_C)
            @test C ≈ h_C
            @test_throws DimensionMismatch CUBLAS.syr2k!('U','N',alpha,d_A,d_Bbad,beta,d_C)
        end
        @testset "syr2k" begin
            alpha = rand(elty)
            A = rand(elty,m,k)
            B = rand(elty,m,k)
            # move to device
            d_A = CuArray(A)
            d_B = CuArray(B)
            C = alpha*(A*transpose(B) + B*transpose(A))
            d_C = CUBLAS.syr2k('U','N',alpha,d_A,d_B)
            # move back to host and compare
            C = triu(C)
            h_C = Array(d_C)
            h_C = triu(h_C)
            @test C ≈ h_C
            C = (A*transpose(B) + B*transpose(A))
            d_C = CUBLAS.syr2k('U','N',d_A,d_B)
            C = triu(C)
            h_C = Array(d_C)
            h_C = triu(h_C)
            @test C ≈ h_C
        end
        @testset "diagm" begin
            A = rand(elty, m)
            B = rand(elty, n)
            # move to device
            d_A = CuArray(A)
            d_B = CuArray(B)
            diagA = diagm(d_A)
            diagB = diagm(2 => d_B)
            # move back to host and compare
            diagind_A = diagind(diagA, 0)
            diagind_B = diagind(diagB, 2)
            h_A = Array(diagA[diagind_A])
            h_B = Array(diagB[diagind_B])

            @test A ≈ h_A
            @test B ≈ h_B

            diagA = diagm(m, m, 0 => d_A)
            diagind_A = diagind(diagA, 0)
            h_A = Array(diagA[diagind_A])
            @test A ≈ h_A

            diagA = diagm(m, m, d_A)
            diagind_A = diagind(diagA, 0)
            h_A = Array(diagA[diagind_A])
            @test A ≈ h_A
        end
        if elty <: Complex
            @testset "herk!" begin
                alpha = rand(real(elty))
                beta = rand(real(elty))
                A = rand(elty,m,m)
                hA = A + A'
                d_A = CuArray(A)
                d_C = CuArray(hA)
                CUBLAS.herk!('U','N',alpha,d_A,beta,d_C)
                C = real(alpha)*(A*A') + real(beta)*hA
                C = triu(C)
                # move to host and compare
                h_C = Array(d_C)
                h_C = triu(C)
                @test C ≈ h_C
            end
            @testset "herk" begin
                A = rand(elty,m,m)
                d_A = CuArray(A)
                d_C = CUBLAS.herk('U','N',d_A)
                C = A*A'
                C = triu(C)
                # move to host and compare
                h_C = Array(d_C)
                h_C = triu(C)
                @test C ≈ h_C
            end
            @testset "her2k!" begin
                elty1 = elty
                elty2 = real(elty)
                # generate parameters
                α = rand(elty1)
                β = rand(elty2)
                A = rand(elty,m,k)
                B = rand(elty,m,k)
                Bbad = rand(elty,m+1,k+1)
                C = rand(elty,m,m)
                # move to device
                d_A = CuArray(A)
                d_B = CuArray(B)
                d_Bbad = CuArray(Bbad)
                C = C + C'
                d_C = CuArray(C)
                C = α*(A*B') + conj(α)*(B*A') + β*C
                CUBLAS.her2k!('U','N',α,d_A,d_B,β,d_C)
                # move back to host and compare
                C = triu(C)
                h_C = Array(d_C)
                h_C = triu(h_C)
                @test C ≈ h_C
                @test_throws DimensionMismatch CUBLAS.her2k!('U','N',α,d_A,d_Bbad,β,d_C)
                Bbad = rand(elty,m,k+1)
                d_Bbad = CuArray(Bbad)
                @test_throws DimensionMismatch CUBLAS.her2k!('U','N',α,d_A,d_Bbad,β,d_C)
            end
            @testset "her2k" begin
                α = rand(elty)
                A = rand(elty,m,k)
                B = rand(elty,m,k)
                d_A = CuArray(A)
                d_B = CuArray(B)
                C = (α*A*B' + conj(α)*B*A')
                d_C = CUBLAS.her2k('U','N',α,d_A,d_B)
                # move back to host and compare
                C = triu(C)
                h_C = Array(d_C)
                h_C = triu(h_C)
                @test C ≈ h_C
                C = (A*B' + B*A')
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
