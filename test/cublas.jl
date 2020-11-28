using CUDA.CUBLAS
using CUDA.CUBLAS: band, bandex

using LinearAlgebra

using BFloat16s

@test CUBLAS.version() isa VersionNumber
@test CUBLAS.version().major == CUBLAS.cublasGetProperty(CUDA.MAJOR_VERSION)
@test CUBLAS.version().minor == CUBLAS.cublasGetProperty(CUDA.MINOR_VERSION)
@test CUBLAS.version().patch == CUBLAS.cublasGetProperty(CUDA.PATCH_LEVEL)

m = 20
n = 35
k = 13

# HACK: remove me when a new version of BFloat16s.jl is released
Base.eps(::Type{BFloat16}) = Base.bitcast(BFloat16, 0x3c00)

############################################################################################

@testset "level 1" begin
    @testset for T in [Float32, Float64, ComplexF32, ComplexF64]
        A = CUDA.rand(T, m)
        B = CuArray{T}(undef, m)
        CUBLAS.copy!(m,A,B)
        @test Array(A) == Array(B)

        @test testf(rmul!, rand(T, 6, 9, 3), Ref(rand()))
        @test testf(dot, rand(T, m), rand(T, m))
        @test testf(*, transpose(rand(T, m)), rand(T, m))
        @test testf(*, rand(T, m)', rand(T, m))
        @test testf(norm, rand(T, m))
        @test testf(BLAS.asum, rand(T, m))
        @test testf(BLAS.axpy!, Ref(rand()), rand(T, m), rand(T, m))
        @test testf(BLAS.axpby!, Ref(rand()), rand(T, m), Ref(rand()), rand(T, m))

        if T <: Complex
            @test testf(BLAS.dotu, rand(T, m), rand(T, m))
            x = rand(T, m)
            y = rand(T, m)
            dx = CuArray(x)
            dy = CuArray(y)
            dz = BLAS.dot(dx, dy)
            z = BLAS.dotc(x, y)
            @test dz ≈ z
        end

        @test testf(rotate!, rand(T, m), rand(T, m), rand(real(T)), rand(real(T)))
        @test testf(rotate!, rand(T, m), rand(T, m), rand(real(T)), rand(T))

        @test testf(reflect!, rand(T, m), rand(T, m), rand(real(T)), rand(real(T)))
        @test testf(reflect!, rand(T, m), rand(T, m), rand(real(T)), rand(T))
    end # level 1 testset
end

############################################################################################

@testset "level 2" begin
    @testset for elty in [Float32, Float64, ComplexF32, ComplexF64]
        alpha = rand(elty)
        beta = rand(elty)

        @testset "gemv" begin
            @test testf(*, rand(elty, m, n), rand(elty, n))
            @test testf(*, transpose(rand(elty, m, n)), rand(elty, m))
            @test testf(*, rand(elty, m, n)', rand(elty, m))
            x = rand(elty, m)
            A = rand(elty, m, m + 1 )
            y = rand(elty, m)
            dx = CuArray(x)
            dA = CuArray(A)
            dy = CuArray(y)
            @test_throws DimensionMismatch mul!(dy, dA, dx)
            A = rand(elty, m + 1, m )
            dA = CuArray(A)
            @test_throws DimensionMismatch mul!(dy, dA, dx)
        end

        @testset "mul! y = $f(A) * x * $Ts(a) + y * $Ts(b)" for f in (identity, transpose, adjoint), Ts in (Int, elty)
            y, A, x = rand(elty, 5), rand(elty, 5, 5), rand(elty, 5)
            dy, dA, dx = CuArray(y), CuArray(A), CuArray(x)
            mul!(dy, f(dA), dx, Ts(1), Ts(2))
            mul!(y, f(A), x, Ts(1), Ts(2))
            @test Array(dy) ≈ y
        end
        @testset "banded methods" begin
            # bands
            ku = 2
            kl = 3
            # generate banded matrix
            A = rand(elty,m,n)
            A = bandex(A,kl,ku)
            # get packed format
            Ab = band(A,kl,ku)
            d_Ab = CuArray(Ab)
            x = rand(elty,n)
            d_x = CuArray(x)
            @testset "gbmv!" begin
                # test y = alpha*A*x + beta*y
                y = rand(elty,m)
                d_y = CuArray(y)
                CUBLAS.gbmv!('N',m,kl,ku,alpha,d_Ab,d_x,beta,d_y)
                BLAS.gbmv!('N',m,kl,ku,alpha,Ab,x,beta,y)
                h_y = Array(d_y)
                @test y ≈ h_y
                # test y = alpha*transpose(A)*x + beta*y
                x = rand(elty,n)
                d_x = CuArray(x)
                y = rand(elty,m)
                d_y = CuArray(y)
                CUBLAS.gbmv!('T',m,kl,ku,alpha,d_Ab,d_y,beta,d_x)
                BLAS.gbmv!('T',m,kl,ku,alpha,Ab,y,beta,x)
                h_x = Array(d_x)
                @test x ≈ h_x
                # test y = alpha*A'*x + beta*y
                x = rand(elty,n)
                d_x = CuArray(x)
                y = rand(elty,m)
                d_y = CuArray(y)
                CUBLAS.gbmv!('C',m,kl,ku,alpha,d_Ab,d_y,beta,d_x)
                BLAS.gbmv!('C',m,kl,ku,alpha,Ab,y,beta,x)
                h_x = Array(d_x)
                @test x ≈ h_x
            end
            x = rand(elty,n)
            d_x = CuArray(x)
            @testset "gbmv" begin
                # test y = alpha*A*x
                d_y = CUBLAS.gbmv('N',m,kl,ku,alpha,d_Ab,d_x)
                y = zeros(elty,m)
                y = BLAS.gbmv('N',m,kl,ku,alpha,Ab,x)
                h_y = Array(d_y)
                @test y ≈ h_y
            end
            A = rand(elty,m,m)
            A = A + A'
            nbands = 3
            @test m >= 1+nbands
            A = bandex(A,nbands,nbands)
            # convert to 'upper' banded storage format
            AB = band(A,0,nbands)
            # construct x
            x = rand(elty,m)
            d_AB = CuArray(AB)
            d_x = CuArray(x)
            if elty <: Real
                @testset "sbmv!" begin
                    y = rand(elty,m)
                    d_y = CuArray(y)
                    # sbmv!
                    CUBLAS.sbmv!('U',nbands,alpha,d_AB,d_x,beta,d_y)
                    y = alpha*(A*x) + beta*y
                    # compare
                    h_y = Array(d_y)
                    @test y ≈ h_y
                end
                @testset "sbmv" begin
                    d_y = CUBLAS.sbmv('U',nbands,d_AB,d_x)
                    y = A*x
                    # compare
                    h_y = Array(d_y)
                    @test y ≈ h_y
                end
            else
                @testset "hbmv!" begin
                    y = rand(elty,m)
                    d_y = CuArray(y)
                    # hbmv!
                    CUBLAS.hbmv!('U',nbands,alpha,d_AB,d_x,beta,d_y)
                    y = alpha*(A*x) + beta*y
                    # compare
                    h_y = Array(d_y)
                    @test y ≈ h_y
                end
                @testset "hbmv" begin
                    d_y = CUBLAS.hbmv('U',nbands,d_AB,d_x)
                    y = A*x
                    # compare
                    h_y = Array(d_y)
                    @test y ≈ h_y
                end
            end
            # generate triangular matrix
            A = rand(elty,m,m)
            # restrict to 3 bands
            nbands = 3
            @test m >= 1+nbands
            A = bandex(A,0,nbands)
            # convert to 'upper' banded storage format
            AB = band(A,0,nbands)
            d_AB = CuArray(AB)
            @testset "tbmv!" begin
                y = rand(elty, m)
                # move to host
                d_y = CuArray(y)
                # tbmv!
                CUBLAS.tbmv!('U','N','N',nbands,d_AB,d_y)
                y = A*y
                # compare
                h_y = Array(d_y)
                @test y ≈ h_y
            end
            @testset "tbmv" begin
                # tbmv
                d_y = CUBLAS.tbmv('U','N','N',nbands,d_AB,d_x)
                y = A*x
                # compare
                h_y = Array(d_y)
                @test y ≈ h_y
            end
            @testset "tbsv!" begin
                d_y = copy(d_x)
                #tbsv!
                CUBLAS.tbsv!('U','N','N',nbands,d_AB,d_y)
                y = A\x
                # compare
                h_y = Array(d_y)
                @test y ≈ h_y
            end
            @testset "tbsv" begin
                d_y = CUBLAS.tbsv('U','N','N',nbands,d_AB,d_x)
                y = A\x
                # compare
                h_y = Array(d_y)
                @test y ≈ h_y
            end
        end
        A = rand(elty,m,n)
        dA = CuArray(A)
        sA = rand(elty,m,m)
        sA = sA + transpose(sA)
        dsA = CuArray(sA)
        hA = rand(elty,m,m)
        hA = hA + hA'
        dhA = CuArray(hA)
        x = rand(elty,m)
        dx = CuArray(x)
        @testset "symv!" begin
            # generate vectors
            y = rand(elty,m)
            # copy to device
            dy = CuArray(y)
            # execute on host
            BLAS.symv!('U',alpha,sA,x,beta,y)
            # execute on device
            CUBLAS.symv!('U',alpha,dsA,dx,beta,dy)
            # compare results
            hy = Array(dy)
            @test y ≈ hy
        end

        @testset "symv" begin
            y = BLAS.symv('U',sA,x)
            # execute on device
            dy = CUBLAS.symv('U',dsA,dx)
            # compare results
            hy = Array(dy)
            @test y ≈ hy
        end
        if elty <: Complex
            @testset "hemv!" begin
                y = rand(elty,m)
                dy = CuArray(y)
                # execute on host
                BLAS.hemv!('U',alpha,hA,x,beta,y)
                # execute on device
                CUBLAS.hemv!('U',alpha,dhA,dx,beta,dy)
                # compare results
                hy = Array(dy)
                @test y ≈ hy
            end
            @testset "hemv" begin
                y = BLAS.hemv('U',hA,x)
                # execute on device
                dy = CUBLAS.hemv('U',dhA,dx)
                # compare results
                hy = Array(dy)
                @test y ≈ hy
            end
        end
        A = triu(sA)
        dA = CuArray(A)
        @testset "trmv!" begin
            d_y = copy(dx)
            # execute trmv!
            CUBLAS.trmv!('U','N','N',dA,d_y)
            y = A*x
            # compare
            h_y = Array(d_y)
            @test y ≈ h_y
        end

        @testset "trmv" begin
            d_y = CUBLAS.trmv('U','N','N',dA,dx)
            y = A*x
            # compare
            h_y = Array(d_y)
            @test y ≈ h_y
        end

        @testset "lmul!(::UpperTriangular)" begin
            dy = copy(dx)
            lmul!(UpperTriangular(dA), dy)
            y = UpperTriangular(A) * x
            @test y ≈ Array(dy)
        end
        @testset "lmul!(::UpperTriangular{Adjoint})" begin
            dy = copy(dx)
            lmul!(adjoint(UpperTriangular(dA)), dy)
            y = adjoint(UpperTriangular(A)) * x
            @test y ≈ Array(dy)
        end
        @testset "lmul!(::UpperTriangular{Transpose})" begin
            dy = copy(dx)
            lmul!(transpose(UpperTriangular(dA)), dy)
            y = transpose(UpperTriangular(A)) * x
            @test y ≈ Array(dy)
        end
        @testset "lmul!(::LowerTriangular)" begin
            dy = copy(dx)
            lmul!(LowerTriangular(dA), dy)
            y = LowerTriangular(A) * x
            @test y ≈ Array(dy)
        end
        @testset "lmul!(::LowerTriangular{Adjoint})" begin
            dy = copy(dx)
            lmul!(adjoint(LowerTriangular(dA)), dy)
            y = adjoint(LowerTriangular(A)) * x
            @test y ≈ Array(dy)
        end
        @testset "lmul!(::LowerTriangular{Transpose})" begin
            dy = copy(dx)
            lmul!(transpose(LowerTriangular(dA)), dy)
            y = transpose(LowerTriangular(A)) * x
            @test y ≈ Array(dy)
        end

        @testset "trsv!" begin
            d_y = copy(dx)
            # execute trsv!
            CUBLAS.trsv!('U','N','N',dA,d_y)
            y = A\x
            # compare
            h_y = Array(d_y)
            @test y ≈ h_y
        end

        @testset "trsv" begin
            d_y = CUBLAS.trsv('U','N','N',dA,dx)
            y = A\x
            # compare
            h_y = Array(d_y)
            @test y ≈ h_y
        end
        @testset "trsv (adjoint)" begin
            d_y = CUBLAS.trsv('U','C','N',dA,dx)
            y = adjoint(A)\x
            # compare
            h_y = Array(d_y)
            @test y ≈ h_y
        end
        @testset "trsv (transpose)" begin
            d_y = CUBLAS.trsv('U','T','N',dA,dx)
            y = transpose(A)\x
            # compare
            h_y = Array(d_y)
            @test y ≈ h_y
        end

        @testset "ldiv!(::UpperTriangular)" begin
            A = copy(sA)
            dA = CuArray(A)
            dy = copy(dx)
            ldiv!(UpperTriangular(dA), dy)
            y = UpperTriangular(A) \ x
            @test y ≈ Array(dy)
        end
        @testset "ldiv!(::UpperTriangular{Adjoint})" begin
            A = copy(sA)
            dA = CuArray(A)
            dy = copy(dx)
            ldiv!(adjoint(UpperTriangular(dA)), dy)
            y = adjoint(UpperTriangular(A)) \ x
            @test y ≈ Array(dy)
        end
        @testset "ldiv!(::UpperTriangular{Transpose})" begin
            A = copy(sA)
            dA = CuArray(A)
            dy = copy(dx)
            ldiv!(transpose(UpperTriangular(dA)), dy)
            y = transpose(UpperTriangular(A)) \ x
            @test y ≈ Array(dy)
        end
        @testset "ldiv!(::LowerTriangular)" begin
            A = copy(sA)
            dA = CuArray(A)
            dy = copy(dx)
            ldiv!(LowerTriangular(dA), dy)
            y = LowerTriangular(A) \ x
            @test y ≈ Array(dy)
        end
        @testset "ldiv!(::LowerTriangular{Adjoint})" begin
            A = copy(sA)
            dA = CuArray(A)
            dy = copy(dx)
            ldiv!(adjoint(LowerTriangular(dA)), dy)
            y = adjoint(LowerTriangular(A)) \ x
            @test y ≈ Array(dy)
        end
        @testset "ldiv!(::LowerTriangular{Transpose})" begin
            A = copy(sA)
            dA = CuArray(A)
            dy = copy(dx)
            ldiv!(transpose(LowerTriangular(dA)), dy)
            y = transpose(LowerTriangular(A)) \ x
            @test y ≈ Array(dy)
        end

        @testset "inv($TR)" for TR in (UpperTriangular, LowerTriangular, UnitUpperTriangular, UnitLowerTriangular)
            @test testf(x -> inv(TR(x)), rand(elty, m, m))
        end

        A = rand(elty,m,m)
        x = rand(elty,m)
        y = rand(elty,m)
        dA = CuArray(A)
        dx = CuArray(x)
        dy = CuArray(y)
        @testset "ger!" begin
            # perform rank one update
            dB = copy(dA)
            CUBLAS.ger!(alpha,dx,dy,dB)
            B = (alpha*x)*y' + A
            # move to host and compare
            hB = Array(dB)
            @test B ≈ hB
        end

        @testset "syr!" begin
            dB = copy(dsA)
            CUBLAS.syr!('U',alpha,dx,dB)
            B = (alpha*x)*transpose(x) + sA
            # move to host and compare upper triangles
            hB = Array(dB)
            B = triu(B)
            hB = triu(hB)
            @test B ≈ hB
        end
        if elty <: Complex
            @testset "her!" begin
                dB = copy(dhA)
                # perform rank one update
                CUBLAS.her!('U',real(alpha),dx,dB)
                B = (real(alpha)*x)*x' + hA
                # move to host and compare upper triangles
                hB = Array(dB)
                B = triu(B)
                hB = triu(hB)
                @test B ≈ hB
            end

            @testset "her2!" begin
                dB = copy(dhA)
                CUBLAS.her2!('U',real(alpha),dx,dy,dB)
                B = (real(alpha)*x)*y' + y*(real(alpha)*x)' + hA
                # move to host and compare upper triangles
                hB = Array(dB)
                B = triu(B)
                hB = triu(hB)
                @test B ≈ hB
            end
        end
    end

    @testset "gemv! with strided inputs" begin  # JuliaGPU/CUDA.jl#445
        testf(rand(16), rand(4)) do p, b
            W = @view p[reshape(1:(16),4,4)]
            W*b
        end
    end
end

############################################################################################

@testset "level 3" begin
    @testset for elty in [Float32, Float64, ComplexF32, ComplexF64]
        alpha = rand(elty)
        beta = rand(elty)

        @testset "mul! C = $f(A) *  $g(B) * $Ts(a) + C * $Ts(b)" for f in (identity, transpose, adjoint), g in (identity, transpose, adjoint), Ts in (Int, elty)
            C, A, B = rand(elty, 5, 5), rand(elty, 5, 5), rand(elty, 5, 5)
            dC, dA, dB = CuArray(C), CuArray(A), CuArray(B)
            mul!(dC, f(dA), g(dB), Ts(1), Ts(2))
            mul!(C, f(A), g(B), Ts(1), Ts(2))
            @test Array(dC) ≈ C
        end
        A = rand(elty,m,k)
        B = rand(elty,k,n)
        Bbad = rand(elty,k+1,n+1)
        C1 = rand(elty,m,n)
        C2 = copy(C1)
        d_A = CuArray(A)
        d_B = CuArray(B)
        d_Bbad = CuArray(Bbad)
        d_C1 = CuArray(C1)
        d_C2 = CuArray(C2)
        hA = rand(elty,m,m)
        hA = hA + hA'
        dhA = CuArray(hA)
        sA = rand(elty,m,m)
        sA = sA + transpose(sA)
        dsA = CuArray(sA)
        @testset "gemm!" begin
            CUBLAS.gemm!('N','N',alpha,d_A,d_B,beta,d_C1)
            mul!(d_C2, d_A, d_B)
            h_C1 = Array(d_C1)
            h_C2 = Array(d_C2)
            C1 = (alpha*A)*B + beta*C1
            C2 = A*B
            # compare
            @test C1 ≈ h_C1
            @test C2 ≈ h_C2
            @test_throws ArgumentError mul!(dhA, dhA, dsA)
            @test_throws DimensionMismatch mul!(d_C1, d_A, dsA)
        end
        if capability(device()) > v"5.0"
            @testset "gemmEx!" begin
                α = rand(elty)
                β = rand(elty)
                CUBLAS.gemmEx!('N','N',α,d_A,d_B,β,d_C1)
                h_C1 = Array(d_C1)
                C1 = (α*A)*B + β*C1
                # compare
                @test C1 ≈ h_C1
            end
        end
        @testset "gemm" begin
            d_C = CUBLAS.gemm('N','N',d_A,d_B)
            C = A*B
            C2 = d_A * d_B
            # compare
            h_C = Array(d_C)
            h_C2 = Array(C2)
            @test C ≈ h_C
            @test C ≈ h_C2
        end
        @testset "xt_gemm! gpu" begin
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
            @test_throws DimensionMismatch CUBLAS.xt_gemm!('N','N',alpha,d_A,d_Bbad,beta,d_C1)
        end
        @testset "xt_gemm! cpu" begin
            h_C1 = Array(d_C1)
            CUBLAS.xt_gemm!('N','N',alpha,Array(d_A),Array(d_B),beta,h_C1)
            mul!(d_C2, d_A, d_B)
            h_C2 = Array(d_C2)
            C1 = (alpha*A)*B + beta*C1
            C2 = A*B
            # compare
            @test C1 ≈ h_C1
            @test C2 ≈ h_C2
        end

        @testset "xt_gemm gpu" begin
            synchronize()
            d_C = CUBLAS.xt_gemm('N','N',d_A,d_B)
            C = A*B
            C2 = d_A * d_B
            # compare
            @test d_C isa CuArray
            h_C = Array(d_C)
            h_C2 = Array(C2)
            @test C ≈ h_C
            @test C ≈ h_C2
        end
        @testset "xt_gemm cpu" begin
            h_C = CUBLAS.xt_gemm('N','N',Array(d_A),Array(d_B))
            C = A*B
            C2 = d_A * d_B
            # compare
            @test h_C isa Array
            h_C2 = Array(C2)
            @test C ≈ h_C
            @test C ≈ h_C2
        end
        # generate matrices
        bA = [rand(elty,m,k) for i in 1:10]
        bB = [rand(elty,k,n) for i in 1:10]
        bC = [rand(elty,m,n) for i in 1:10]
        # move to device
        bd_A = CuArray{elty, 2}[]
        bd_B = CuArray{elty, 2}[]
        bd_C = CuArray{elty, 2}[]
        bd_bad = CuArray{elty, 2}[]
        for i in 1:length(bA)
            push!(bd_A,CuArray(bA[i]))
            push!(bd_B,CuArray(bB[i]))
            push!(bd_C,CuArray(bC[i]))
            if i < length(bA) - 2
                push!(bd_bad,CuArray(bC[i]))
            end
        end
        @testset "gemm_batched!" begin
            # C = (alpha*A)*B + beta*C
            CUBLAS.gemm_batched!('N','N',alpha,bd_A,bd_B,beta,bd_C)
            for i in 1:length(bd_C)
                bC[i] = (alpha*bA[i])*bB[i] + beta*bC[i]
                h_C = Array(bd_C[i])
                #compare
                @test bC[i] ≈ h_C
            end
            @test_throws DimensionMismatch CUBLAS.gemm_batched!('N','N',alpha,bd_A,bd_bad,beta,bd_C)
        end

        @testset "gemm_batched" begin
            bd_C = CUBLAS.gemm_batched('N','N',bd_A,bd_B)
            for i in 1:length(bA)
                bC = bA[i]*bB[i]
                h_C = Array(bd_C[i])
                @test bC ≈ h_C
            end
            @test_throws DimensionMismatch CUBLAS.gemm_batched('N','N',alpha,bd_A,bd_bad)
        end

        nbatch = 10
        bA = rand(elty, m, k, nbatch)
        bB = rand(elty, k, n, nbatch)
        bC = rand(elty, m, n, nbatch)
        bbad = rand(elty, m+1, n+1, nbatch)
        # move to device
        bd_A = CuArray{elty, 3}(bA)
        bd_B = CuArray{elty, 3}(bB)
        bd_C = CuArray{elty, 3}(bC)
        bd_bad = CuArray{elty, 3}(bbad)
        @testset "gemm_strided_batched!" begin
            CUBLAS.gemm_strided_batched!('N', 'N', alpha, bd_A, bd_B, beta, bd_C)
            for i in 1:nbatch
                bC[:, :, i] = (alpha * bA[:, :, i]) * bB[:, :, i] + beta * bC[:, :, i]
            end
            h_C = Array(bd_C)
            @test bC ≈ h_C
            @test_throws DimensionMismatch CUBLAS.gemm_strided_batched!('N', 'N', alpha, bd_A, bd_B, beta, bd_bad)
        end

        @testset "gemm_strided_batched" begin
            bd_C = CUBLAS.gemm_strided_batched('N', 'N', bd_A, bd_B)

            for i in 1:nbatch
                bC[:, :, i] = bA[:, :, i] * bB[:, :, i]
            end
            h_C = Array(bd_C)
            @test bC ≈ h_C
            # generate matrices
            bA = rand(elty, k, m, nbatch)
            bB = rand(elty, k, n, nbatch)
            bC = zeros(elty, m, n, nbatch)
            # move to device
            bd_A = CuArray{elty, 3}(bA)
            bd_B = CuArray{elty, 3}(bB)

            bd_C = CUBLAS.gemm_strided_batched('T', 'N', bd_A, bd_B)
            for i in 1:nbatch
                bC[:, :, i] = transpose(bA[:, :, i]) * bB[:, :, i]
            end
            h_C = Array(bd_C)
            @test bC ≈ h_C
            @test_throws DimensionMismatch CUBLAS.gemm_strided_batched('N', 'N', alpha, bd_A, bd_bad)
        end

        B = rand(elty,m,n)
        C = rand(elty,m,n)
        Bbad = rand(elty,m+1,n+1)
        d_B = CuArray(B)
        d_C = CuArray(C)
        d_Bbad = CuArray(Bbad)
        @testset "symm!" begin
            CUBLAS.symm!('L','U',alpha,dsA,d_B,beta,d_C)
            C = (alpha*sA)*B + beta*C
            # compare
            h_C = Array(d_C)
            @test C ≈ h_C
            @test_throws DimensionMismatch CUBLAS.symm!('L','U',alpha,dsA,d_Bbad,beta,d_C)
        end

        @testset "symm" begin
            d_C = CUBLAS.symm('L','U',dsA,d_B)
            C = sA*B
            # compare
            h_C = Array(d_C)
            @test C ≈ h_C
            @test_throws DimensionMismatch CUBLAS.symm('L','U',dsA,d_Bbad)
        end
        @testset "xt_symm! gpu" begin
            synchronize()
            CUBLAS.xt_symm!('L','U',alpha,dsA,d_B,beta,d_C)
            C = (alpha*sA)*B + beta*C
            # compare
            h_C = Array(d_C)
            @test C ≈ h_C
        end
        @testset "xt_symm! cpu" begin
            h_C = Array(d_C)
            CUBLAS.xt_symm!('L','U',alpha,Array(dsA),Array(d_B),beta,h_C)
            C = (alpha*sA)*B + beta*C
            # compare
            @test C ≈ h_C
        end

        @testset "xt_symm gpu" begin
            synchronize()
            d_C = CUBLAS.xt_symm('L','U',dsA,d_B)
            C = sA*B
            # compare
            @test d_C isa CuArray
            h_C = Array(d_C)
            @test C ≈ h_C
        end
        @testset "xt_symm cpu" begin
            h_C = CUBLAS.xt_symm('L','U',Array(dsA),Array(d_B))
            C = sA*B
            # compare
            @test h_C isa Array
            @test C ≈ h_C
        end
        A = triu(rand(elty, m, m))
        B = rand(elty,m,n)
        C = zeros(elty,m,n)
        dA = CuArray(A)
        dB = CuArray(B)
        dC = CuArray(C)
        @testset "trmm!" begin
            C = alpha*A*B
            CUBLAS.trmm!('L','U','N','N',alpha,dA,dB,dC)
            # move to host and compare
            h_C = Array(dC)
            @test C ≈ h_C
        end
        @testset "trmm" begin
            C = alpha*A*B
            d_C = CUBLAS.trmm('L','U','N','N',alpha,dA,dB)
            # move to host and compare
            h_C = Array(d_C)
            @test C ≈ h_C
        end
        @testset "xt_trmm! gpu" begin
            C = alpha*A*B
            synchronize()
            CUBLAS.xt_trmm!('L','U','N','N',alpha,dA,dB,dC)
            # move to host and compare
            h_C = Array(dC)
            @test C ≈ h_C
        end
        @testset "xt_trmm! cpu" begin
            C = alpha*A*B
            h_C = Array(dC)
            CUBLAS.xt_trmm!('L','U','N','N',alpha,Array(dA),Array(dB),h_C)
            @test C ≈ h_C
        end
        @testset "xt_trmm gpu" begin
            C = alpha*A*B
            synchronize()
            d_C = CUBLAS.xt_trmm('L','U','N','N',alpha,dA,dB)
            # move to host and compare
            @test d_C isa CuArray
            h_C = Array(d_C)
            @test C ≈ h_C
        end
        @testset "xt_trmm cpu" begin
            C = alpha*A*B
            h_C = CUBLAS.xt_trmm('L','U','N','N',alpha,Array(dA),Array(dB))
            @test h_C isa Array
            @test C ≈ h_C
        end

        @testset "xt_trsm! gpu" begin
            C = alpha*(A\B)
            dC = copy(dB)
            synchronize()
            CUBLAS.xt_trsm!('L','U','N','N',alpha,dA,dC)
            # move to host and compare
            h_C = Array(dC)
            @test C ≈ h_C
        end
        @testset "xt_trsm! cpu" begin
            C = alpha*(A\B)
            dC = copy(dB)
            h_C = Array(dC)
            CUBLAS.xt_trsm!('L','U','N','N',alpha,Array(dA),h_C)
            @test C ≈ h_C
        end
        @testset "xt_trsm gpu" begin
            C  = alpha*(A\B)
            synchronize()
            dC = CUBLAS.xt_trsm('L','U','N','N',alpha,dA,dB)
            # move to host and compare
            @test dC isa CuArray
            h_C = Array(dC)
            @test C ≈ h_C
        end
        @testset "xt_trsm cpu" begin
            C  = alpha*(A\B)
            h_C = CUBLAS.xt_trsm('L','U','N','N',alpha,Array(dA),Array(dB))
            @test h_C isa Array
            @test C ≈ h_C
        end
        @testset "trsm" begin
            Br = rand(elty,m,n)
            Bl = rand(elty,n,m)
            d_Br = CuArray(Br)
            d_Bl = CuArray(Bl)
            # compute
            @testset "adjtype=$adjtype, uplotype=$uplotype" for
                adjtype in (identity, adjoint, transpose),
                    uplotype in (UpperTriangular, UnitUpperTriangular, LowerTriangular, UnitLowerTriangular)

                @test adjtype(uplotype(A))\Br ≈ Array(adjtype(uplotype(dA))\d_Br)
                @test Bl/adjtype(uplotype(A)) ≈ Array(d_Bl/adjtype(uplotype(dA)))
            end
            # Check also that scaling parameter works
            @test BLAS.trsm('L','U','N','N',alpha,A,Br) ≈ Array(CUBLAS.trsm('L','U','N','N',alpha,dA,d_Br))
        end

        @testset "trsm_batched!" begin
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
            # generate parameter alpha = rand(elty)
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

        @testset "triangular mul!" begin
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

        B = rand(elty,m,n)
        C = rand(elty,m,n)
        d_B = CuArray(B)
        d_C = CuArray(C)
        if elty <: Complex
            @testset "hemm!" begin
                # compute
                C = alpha*(hA*B) + beta*C
                CUBLAS.hemm!('L','L',alpha,dhA,d_B,beta,d_C)
                # move to host and compare
                h_C = Array(d_C)
                @test C ≈ h_C
            end
            @testset "hemm" begin
                C = hA*B
                d_C = CUBLAS.hemm('L','U',dhA,d_B)
                # move to host and compare
                h_C = Array(d_C)
                @test C ≈ h_C
            end
            @testset "xt_hemm! gpu" begin
                # compute
                C = alpha*(hA*B) + beta*C
                synchronize()
                CUBLAS.xt_hemm!('L','L',alpha,dhA,d_B,beta,d_C)
                # move to host and compare
                h_C = Array(d_C)
                @test C ≈ h_C
            end
            @testset "xt_hemm! cpu" begin
                # compute
                C = alpha*(hA*B) + beta*C
                h_C = Array(d_C)
                CUBLAS.xt_hemm!('L','L',alpha,Array(dhA),Array(d_B),beta,h_C)
                @test C ≈ h_C
            end
            @testset "xt_hemm gpu" begin
                C   = hA*B
                synchronize()
                d_C = CUBLAS.xt_hemm('L','U',dhA, d_B)
                # move to host and compare
                @test d_C isa CuArray
                h_C = Array(d_C)
                @test C ≈ h_C
            end
            @testset "xt_hemm cpu" begin
                C   = hA*B
                h_C = CUBLAS.xt_hemm('L','U',Array(dhA), Array(d_B))
                # move to host and compare
                @test h_C isa Array
                @test C ≈ h_C
            end
        end
        A = rand(elty,m,n)
        d_A = CuArray(A)
        @testset "geam!" begin
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
            D = alpha*A + beta*B
            d_C = CUBLAS.geam('N','N',alpha,d_A,beta,d_B)
            # move to host and compare
            h_C = Array(d_C)
            @test D ≈ h_C
        end
        A = rand(elty,m,k)
        d_A = CuArray(A)
        @testset "syrkx!" begin
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
            badC = rand(elty, m, n)
            d_badC = CuArray(badC)
            @test_throws DimensionMismatch CUBLAS.syrkx!('U','N',alpha,d_syrkx_A,d_syrkx_B,beta,d_badC)
            badC = rand(elty, n+1, n+1)
            d_badC = CuArray(badC)
            @test_throws DimensionMismatch CUBLAS.syrkx!('U','N',alpha,d_syrkx_A,d_syrkx_B,beta,d_badC)
        end
        @testset "xt_syrkx! gpu" begin
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
        @testset "syrk" begin
            # C = A*transpose(A)
            d_C = CUBLAS.syrk('U','N',d_A)
            C = A*transpose(A)
            C = triu(C)
            # move to host and compare
            h_C = Array(d_C)
            h_C = triu(C)
            @test C ≈ h_C
        end
        @testset "xt_syrk gpu" begin
            # C = A*transpose(A)
            synchronize()
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
            # C = A*transpose(A)
            h_C = CUBLAS.xt_syrk('U','N',Array(d_A))
            C = A*transpose(A)
            C = triu(C)
            # move to host and compare
            @test h_C isa Array
            h_C = triu(C)
            @test C ≈ h_C
        end
        if elty <: Complex
            @testset "herk!" begin
                d_C = CuArray(dhA)
                CUBLAS.herk!('U','N',real(alpha),d_A,real(beta),d_C)
                C = real(alpha)*(A*A') + real(beta)*hA
                C = triu(C)
                # move to host and compare
                h_C = Array(d_C)
                h_C = triu(C)
                @test C ≈ h_C
            end
            @testset "herk" begin
                d_C = CUBLAS.herk('U','N',d_A)
                C = A*A'
                C = triu(C)
                # move to host and compare
                h_C = Array(d_C)
                h_C = triu(C)
                @test C ≈ h_C
            end
            @testset "xt_herk! gpu" begin
                d_C = CuArray(dhA)
                C = real(alpha)*(A*A') + real(beta)*Array(d_C)
                synchronize()
                CUBLAS.xt_herk!('U','N',real(alpha),d_A,real(beta),d_C)
                C = triu(C)
                # move to host and compare
                h_C = Array(d_C)
                h_C = triu(h_C)
                @test C ≈ h_C
            end
            @testset "xt_herk! cpu" begin
                h_C = Array(dhA)
                CUBLAS.xt_herk!('U','N',real(alpha),Array(d_A),real(beta),h_C)
                C = real(alpha)*(A*A') + real(beta)*Array(dhA)
                C = triu(C)
                # move to host and compare
                h_C = triu(h_C)
                @test C ≈ h_C
            end
            @testset "xt_herk gpu" begin
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
                h_C = CUBLAS.xt_herk('U','N',Array(d_A))
                C = A*A'
                C = triu(C)
                # move to host and compare
                @test h_C isa Array
                h_C = triu(h_C)
                @test C ≈ h_C
            end
        end
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
        @testset "syr2k!" begin
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
            C = alpha*(A*transpose(B) + B*transpose(A))
            d_C = CUBLAS.syr2k('U','N',alpha,d_A,d_B)
            # move back to host and compare
            C = triu(C)
            h_C = Array(d_C)
            h_C = triu(h_C)
            @test C ≈ h_C
        end
        if elty <: Complex
            @testset "her2k!" begin
                elty1 = elty
                elty2 = real(elty)
                # generate parameters
                α = rand(elty1)
                β = rand(elty2)
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
            end

            @testset "her2k" begin
                C = A*B' + B*A'
                d_C = CUBLAS.her2k('U','N',d_A,d_B)
                # move back to host and compare
                C = triu(C)
                h_C = Array(d_C)
                h_C = triu(h_C)
                @test C ≈ h_C
            end
            @testset "xt_her2k! gpu" begin
                elty1 = elty
                elty2 = real(elty)
                # generate parameters
                α = rand(elty1)
                β = rand(elty2)
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

    @testset "mixed-precision matmul" begin
        m,k,n = 4,4,4
        cudaTypes = (Float16, Complex{Float16}, BFloat16, Complex{BFloat16}, Float32, Complex{Float32},
                    Float64, Complex{Float64}, Int8, Complex{Int8}, UInt8, Complex{UInt8},
                    Int16, Complex{Int16}, UInt16, Complex{UInt16}, Int32, Complex{Int32},
                    UInt32, Complex{UInt32}, Int64, Complex{Int64}, UInt64, Complex{UInt64})

        for AT in cudaTypes, CT in cudaTypes
            BT = AT # gemmEx requires identical A and B types

            # we only test combinations of types that are supported by gemmEx
            if CUBLAS.gemmExComputeType(AT, BT, CT, m,k,n) !== nothing
                A = AT <: BFloat16 ? AT.(rand(m,k)) : rand(AT, m,k)
                B = BT <: BFloat16 ? BT.(rand(k,n)) : rand(BT, k,n)
                C = similar(B, CT)
                mul!(C, A, B)

                # Base can't do Int8*Int8 without losing accuracy
                if (AT == Int8 && BT == Int8) || (AT == Complex{Int8} && BT == Complex{Int8})
                    C = CT.(A) * CT.(B)
                end

                dA = CuArray(A)
                dB = CuArray(B)
                dC = similar(dB, CT)
                mul!(dC, dA, dB)

                rtol = Base.rtoldefault(AT, BT, 0)
                @test C ≈ Array(dC) rtol=rtol
            end
        end

        # also test an unsupported combination (falling back to GPUArrays)
        let AT=BFloat16, BT=Int32, CT=Float64
            A = AT.(rand(m,k))
            B = rand(BT, k,n)
            C = similar(B, CT)
            mul!(C, A, B)

            dA = CuArray(A)
            dB = CuArray(B)
            dC = similar(dB, CT)
            mul!(dC, dA, dB)

            rtol = Base.rtoldefault(AT, BT, 0)
            @test C ≈ Array(dC) rtol=rtol
        end
    end

    @testset "gemm! with strided inputs" begin # JuliaGPU/CUDA.jl#78
        inn = 784; out = 32
        testf(randn(784*100), rand(Float32, 784, 100)) do p, x
            p[reshape(1:(out*inn),out,inn)] * x
            @view(p[reshape(1:(out*inn),out,inn)]) * x
        end
    end
end

############################################################################################

@testset "extensions" begin
    @testset for elty in [Float32, Float64, ComplexF32, ComplexF64]
        @testset "getrf_batched!" begin
            Random.seed!(1)
            local k
            # generate matrices
            A = [rand(elty,m,m) for i in 1:10]
            # move to device
            d_A = CuArray{elty, 2}[]
            for i in 1:length(A)
                push!(d_A,CuArray(A[i]))
            end
            pivot, info = CUBLAS.getrf_batched!(d_A, false)
            h_info = Array(info)
            for As in 1:length(d_A)
                C   = lu!(copy(A[As]), Val(false)) # lu(A[As],pivot=false)
                h_A = Array(d_A[As])
                #reconstruct L,U
                dL = Matrix(one(elty)*I, m, m)
                dU = zeros(elty,(m,m))
                k = h_info[As]
                if( k >= 0 )
                    dL += tril(h_A,-k-1)
                    dU += triu(h_A,k)
                end
                #compare
                @test C.L ≈ dL rtol=1e-2
                @test C.U ≈ dU rtol=1e-2
            end
            for i in 1:length(A)
                d_A[ i ] = CuArray(A[i])
            end
            pivot, info = CUBLAS.getrf_batched!(d_A, true)
            h_info = Array(info)
            h_pivot = Array(pivot)
            for As in 1:length(d_A)
                C   = lu(A[As])
                h_A = Array(d_A[As])
                #reconstruct L,U
                dL = Matrix(one(elty)*I, m, m)
                dU = zeros(elty,(m,m))
                k = h_info[As]
                if( k >= 0 )
                    dL += tril(h_A,-k-1)
                    dU += triu(h_A,k)
                end
                #compare pivots
                @test length(setdiff(h_pivot[:,As],C.p)) == 0
                #make device pivot matrix
                P = Matrix(1.0*I, m, m)
                for row in 1:m
                    temp = copy(P[row,:])
                    P[row,:] = P[h_pivot[row,As],:]
                    P[h_pivot[row,As],:] = temp
                end
                @test inv(P)*dL*dU ≈ inv(C.P) * C.L * C.U
            end
        end

        @testset "getrf_batched" begin
            local k
            # generate matrices
            A = [rand(elty,m,m) for i in 1:10]
            # move to device
            d_A = CuArray{elty, 2}[]
            for i in 1:length(A)
                push!(d_A,CuArray(A[i]))
            end
            pivot, info, d_B = CUBLAS.getrf_batched(d_A, false)
            h_info = Array(info)
            for Bs in 1:length(d_B)
                C   = lu!(copy(A[Bs]),Val(false)) # lu(A[Bs],pivot=false)
                h_B = Array(d_B[Bs])
                #reconstruct L,U
                dL = Matrix(one(elty)*I, m, m)
                dU = zeros(elty,(m,m))
                k = h_info[Bs]
                if( h_info[Bs] >= 0 )
                    dU += triu(h_B,k)
                    dL += tril(h_B,-k-1)
                end
                #compare
                @test C.L ≈ dL rtol=1e-2
                @test C.U ≈ dU rtol=1e-2
            end
        end

        @testset "getrf_strided_batched!" begin
            Random.seed!(1)
            local k
            # generate strided matrix
            A = rand(elty,m,m,10)
            # move to device
            d_A = CuArray(A)
            pivot, info = CUBLAS.getrf_strided_batched!(d_A, false)
            h_info = Array(info)
            for As in 1:size(d_A, 3)
                C   = lu!(copy(A[:,:,As]), Val(false)) # lu(A[As],pivot=false)
                h_A = Array(d_A[:,:,As])
                #reconstruct L,U
                dL = Matrix(one(elty)*I, m, m)
                dU = zeros(elty,(m,m))
                k = h_info[As]
                if( k >= 0 )
                    dL += tril(h_A,-k-1)
                    dU += triu(h_A,k)
                end
                #compare
                @test C.L ≈ dL rtol=1e-2
                @test C.U ≈ dU rtol=1e-2
            end
            d_A = CuArray(A)
            pivot, info = CUBLAS.getrf_strided_batched!(d_A, true)
            h_info = Array(info)
            h_pivot = Array(pivot)
            for As in 1:size(d_A, 3)
                C   = lu(A[:,:,As])
                h_A = Array(d_A[:,:,As])
                #reconstruct L,U
                dL = Matrix(one(elty)*I, m, m)
                dU = zeros(elty,(m,m))
                k = h_info[As]
                if( k >= 0 )
                    dL += tril(h_A,-k-1)
                    dU += triu(h_A,k)
                end
                #compare pivots
                @test length(setdiff(h_pivot[:,As],C.p)) == 0
                #make device pivot matrix
                P = Matrix(1.0*I, m, m)
                for row in 1:m
                    temp = copy(P[row,:])
                    P[row,:] = P[h_pivot[row,As],:]
                    P[h_pivot[row,As],:] = temp
                end
                @test inv(P)*dL*dU ≈ inv(C.P) * C.L * C.U
            end
        end

        @testset "getrf_strided_batched" begin
            local k
            # generate strided matrix
            A = rand(elty,m,m,10)
            # move to device
            d_A = CuArray(A)
            pivot, info, d_B = CUBLAS.getrf_strided_batched(d_A, false)
            h_info = Array(info)
            for Bs in 1:size(d_B, 3)
                C   = lu!(copy(A[:,:,Bs]),Val(false)) # lu(A[Bs],pivot=false)
                h_B = Array(d_B[:,:,Bs])
                #reconstruct L,U
                dL = Matrix(one(elty)*I, m, m)
                dU = zeros(elty,(m,m))
                k = h_info[Bs]
                if( h_info[Bs] >= 0 )
                    dU += triu(h_B,k)
                    dL += tril(h_B,-k-1)
                end
                #compare
                @test C.L ≈ dL rtol=1e-2
                @test C.U ≈ dU rtol=1e-2
            end
        end

        @testset "getri_batched" begin
            # generate matrices
            A = [rand(elty,m,m) for i in 1:10]
            # move to device
            d_A = CuArray{elty, 2}[]
            for i in 1:length(A)
                push!(d_A,CuArray(A[i]))
            end
            pivot, info = CUBLAS.getrf_batched!(d_A, true)
            h_info = Array(info)
            for Cs in 1:length(h_info)
                @test h_info[Cs] == 0
            end
            pivot, info, d_C = CUBLAS.getri_batched(d_A, pivot)
            h_info = Array(info)
            for Cs in 1:length(d_C)
                C   = inv(A[Cs])
                h_C = Array(d_C[Cs])
                @test h_info[Cs] == 0
                @test C ≈ h_C rtol=1e-2
            end
        end

        @testset "matinv_batched" begin
            # generate matrices
            A = [rand(elty,m,m) for i in 1:10]
            # move to device
            d_A = CuArray{elty, 2}[]
            for i in 1:length(A)
                push!(d_A,CuArray(A[i]))
            end
            info, d_C = CUBLAS.matinv_batched(d_A)
            for Cs in 1:length(d_C)
                C   = inv(A[Cs])
                h_C = Array(d_C[Cs])
                @test C ≈ h_C
            end
            push!(d_A, CUDA.rand(elty, m, m+1))
            @test_throws DimensionMismatch CUBLAS.matinv_batched(d_A)
        end

        @testset "geqrf_batched!" begin
            # generate matrices
            A = [rand(elty,m,n) for i in 1:10]
            # move to device
            d_A = CuArray{elty, 2}[]
            for i in 1:length(A)
                push!(d_A,CuArray(A[i]))
            end
            tau, d_A = CUBLAS.geqrf_batched!(d_A)
            for As in 1:length(d_A)
                C   = qr(A[As])
                h_A = Array(d_A[As])
                h_tau = Array(tau[As])
                # build up Q
                Q = Matrix(one(elty)*I, min(m,n), min(m,n))
                for i in 1:min(m,n)
                    v = zeros(elty,m)
                    v[i] = one(elty)
                    v[i+1:m] = h_A[i+1:m,i]
                    Q *= I - h_tau[i] * v * v'
                end
                @test Q ≈ Array(C.Q) rtol=1e-2
            end
        end

        @testset "geqrf_batched" begin
            # generate matrices
            A = [rand(elty,m,n) for i in 1:10]
            # move to device
            d_A = CuArray{elty, 2}[]
            for i in 1:length(A)
                push!(d_A,CuArray(A[i]))
            end
            tau, d_B = CUBLAS.geqrf_batched!(d_A)
            for Bs in 1:length(d_B)
                C   = qr(A[Bs])
                h_B = Array(d_B[Bs])
                h_tau = Array(tau[Bs])
                # build up Q
                Q = Matrix(one(elty)*I, min(m,n), min(m,n))
                for i in 1:min(m,n)
                    v = zeros(elty,m)
                    v[i] = one(elty)
                    v[i+1:m] = h_B[i+1:m,i]
                    Q *= I - h_tau[i] * v * v'
                end
                @test Q ≈ Array(C.Q) rtol=1e-2
            end
        end

        @testset "gels_batched!" begin
            # generate matrices
            A = [rand(elty,n,k) for i in 1:10]
            C = [rand(elty,n,k) for i in 1:10]
            # move to device
            d_A = CuArray{elty, 2}[]
            d_C = CuArray{elty, 2}[]
            for i in 1:length(A)
                push!(d_A,CuArray(A[i]))
                push!(d_C,CuArray(C[i]))
            end
            d_A, d_C, info = CUBLAS.gels_batched!('N',d_A, d_C)
            for Cs in 1:length(d_C)
                X = A[Cs]\C[Cs]
                h_C = Array(d_C[Cs])[1:k,1:k]
                @test X ≈ h_C rtol=1e-2
            end
            push!(d_C,CUDA.rand(elty,n,k-1))
            @test_throws DimensionMismatch CUBLAS.gels_batched!('N',d_A, d_C)
            push!(d_A,CUDA.rand(elty,n,k-1))
            @test_throws DimensionMismatch CUBLAS.gels_batched!('N',d_A, d_C)
            A = [rand(elty,k-1,k) for i in 1:10]
            C = [rand(elty,k-1,k) for i in 1:10]
            # move to device
            d_A = CuArray{elty, 2}[]
            d_C = CuArray{elty, 2}[]
            for i in 1:length(A)
                push!(d_A,CuArray(A[i]))
                push!(d_C,CuArray(C[i]))
            end
            # system is now not overdetermined
            @test_throws ArgumentError CUBLAS.gels_batched!('N',d_A, d_C)
        end

        @testset "gels_batched" begin
            # generate matrices
            A = [rand(elty,n,k) for i in 1:10]
            C = [rand(elty,n,k) for i in 1:10]
            # move to device
            d_A = CuArray{elty, 2}[]
            d_C = CuArray{elty, 2}[]
            for i in 1:length(A)
                push!(d_A,CuArray(A[i]))
                push!(d_C,CuArray(C[i]))
            end
            d_B, d_D, info = CUBLAS.gels_batched('N',d_A, d_C)
            for Ds in 1:length(d_D)
                X = A[Ds]\C[Ds]
                h_D = Array(d_D[Ds])[1:k,1:k]
                @test X ≈ h_D rtol=1e-2
            end
        end
        # generate matrices
        A = rand(elty,m,n)
        C = rand(elty,m,n)
        x = rand(elty,m)
        # move to device
        d_A = CuArray(A)
        d_C = CuArray(C)
        d_x = CuArray(x)
        C = diagm(0 => x) * A
        @testset "dgmm!" begin
            # compute
            CUBLAS.dgmm!('L', d_A, d_x, d_C)
            # move to host and compare
            h_C = Array(d_C)
            @test C ≈ h_C
            # bounds checking
            @test_throws DimensionMismatch CUBLAS.dgmm!('R', d_A, d_x, d_C)
            E = rand(elty,m,m)
            d_E = CuArray(E)
            @test_throws DimensionMismatch CUBLAS.dgmm!('L', d_E, d_x, d_C)
        end
        @testset "dgmm" begin
            d_C = CUBLAS.dgmm('L', d_A, d_x)
            # move to host and compare
            h_C = Array(d_C)
            @test C ≈ h_C
        end
    end # extensions
end # elty
