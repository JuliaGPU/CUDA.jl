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

@testset "level 2" begin
    @testset for elty in [Float32, Float64, ComplexF32, ComplexF64]
        @testset "gemv" begin
            alpha = rand(elty)
            beta = rand(elty)
            @test testf(*, rand(elty, m, n), rand(elty, n))
            @test testf(*, transpose(rand(elty, m, n)), rand(elty, m))
            @test testf(*, rand(elty, m, n)', rand(elty, m))
            x = rand(elty, m)
            A = rand(elty, m, m + 1)
            y = rand(elty, n)
            dx = CuArray(x)
            dA = CuArray(A)
            dy = CuArray(y)
            @test_throws DimensionMismatch mul!(dy, dA, dx)
            A = rand(elty, m + 1, m )
            dA = CuArray(A)
            @test_throws DimensionMismatch mul!(dy, dA, dx)
            x = rand(elty, m)
            A = rand(elty, n, m)
            dx = CuArray(x)
            dA = CuArray(A)
            alpha = rand(elty)
            dy = CUBLAS.gemv('N', alpha, dA, dx)
            hy = collect(dy)
            @test hy ≈ alpha * A * x
            dy = CUBLAS.gemv('N', dA, dx)
            hy = collect(dy)
            @test hy ≈ A * x
            dy = CuArray(y)
            dx = CUBLAS.gemv(elty <: Real ? 'T' : 'C', alpha, dA, dy)
            hx = collect(dx)
            @test hx ≈ alpha * A' * y
        end

        if CUBLAS.version() >= v"11.9"
            @testset "gemv_batched" begin
                alpha = rand(elty)
                beta = rand(elty)
                x = [rand(elty, m) for i=1:10]
                A = [rand(elty, n, m) for i=1:10]
                y = [rand(elty, n) for i=1:10]
                dx = CuArray{elty, 1}[]
                dA = CuArray{elty, 2}[]
                dy = CuArray{elty, 1}[]
                dbad = CuArray{elty, 1}[]
                for i=1:length(A)
                    push!(dA, CuArray(A[i]))
                    push!(dx, CuArray(x[i]))
                    push!(dy, CuArray(y[i]))
                    if i < length(A) - 2
                        push!(dbad,CuArray(dx[i]))
                    end
                end
                @test_throws DimensionMismatch CUBLAS.gemv_batched!('N', alpha, dA, dx, beta, dbad)
                CUBLAS.gemv_batched!('N', alpha, dA, dx, beta, dy)
                for i=1:length(A)
                    hy = collect(dy[i])
                    y[i] = alpha * A[i] * x[i] + beta * y[i]
                    @test y[i] ≈ hy
                end
                dy = CuArray{elty, 1}[]
                for i=1:length(A)
                    push!(dy, CuArray(y[i]))
                end
                CUBLAS.gemv_batched!(elty <: Real ? 'T' : 'C', alpha, dA, dy, beta, dx)
                for i in 1:length(A)
                    hx = collect(dx[i])
                    x[i] = alpha * A[i]' * y[i] + beta * x[i]
                    @test x[i] ≈ hx
                end
            end
        end

        if CUBLAS.version() >= v"11.9"
            @testset "gemv_strided_batched" begin
                alpha = rand(elty)
                beta = rand(elty)
                x = rand(elty, m, 10)
                A = rand(elty, n, m, 10)
                y = rand(elty, n, 10)
                bad = rand(elty, m, 10)
                dx = CuArray(x)
                dA = CuArray(A)
                dy = CuArray(y)
                dbad = CuArray(bad)
                @test_throws DimensionMismatch CUBLAS.gemv_strided_batched!('N', alpha, dA, dx, beta, dbad)
                bad = rand(elty, n, 2)
                dbad = CuArray(bad)
                @test_throws DimensionMismatch CUBLAS.gemv_strided_batched!('N', alpha, dA, dx, beta, dbad)
                CUBLAS.gemv_strided_batched!('N', alpha, dA, dx, beta, dy)
                for i in 1:size(A, 3)
                    hy = collect(dy[:, i])
                    y[:, i] = alpha * A[:, :, i] * x[:, i] + beta * y[:, i]
                    @test y[:, i] ≈ hy
                end
                dy = CuArray(y)
                CUBLAS.gemv_strided_batched!(elty <: Real ? 'T' : 'C', alpha, dA, dy, beta, dx)
                for i in 1:size(A, 3)
                    hx = collect(dx[:, i])
                    x[:, i] = alpha * A[:, :, i]' * y[:, i] + beta * x[:, i]
                    @test x[:, i] ≈ hx
                end
            end
        end

        @testset "mul! y = $f(A) * x * $Ts(a) + y * $Ts(b)" for f in (identity, transpose, adjoint), Ts in (Int, elty)
            y, A, x = rand(elty, 5), rand(elty, 5, 5), rand(elty, 5)
            dy, dA, dx = CuArray(y), CuArray(A), CuArray(x)
            mul!(dy, f(dA), dx, Ts(1), Ts(2))
            mul!(y, f(A), x, Ts(1), Ts(2))
            @test Array(dy) ≈ y
        end

        @testset "hermitian" begin
            y, A, x = rand(elty, 5), Hermitian(rand(elty, 5, 5)), rand(elty, 5)
            dy, dA, dx = CuArray(y), Hermitian(CuArray(A)), CuArray(x)
            mul!(dy, dA, dx)
            mul!(y, A, x)
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
                alpha = rand(elty)
                beta = rand(elty)
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
                # test alpha=1 version without y
                d_y = CUBLAS.gbmv('N',m,kl,ku,d_Ab,d_x)
                y   = BLAS.gbmv('N',m,kl,ku,Ab,x)
                h_y = Array(d_y)
                @test y ≈ h_y
            end
            @testset "gbmv" begin
                x = rand(elty,n)
                d_x = CuArray(x)
                alpha = rand(elty)
                beta = rand(elty)
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
                    alpha = rand(elty)
                    beta = rand(elty)
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
                    alpha = rand(elty)
                    beta = rand(elty)
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

        function pack(A, uplo)
            AP = Vector{elty}(undef, (n*(n+1))>>1)
            k = 1
            for j in 1:n
                for i in (uplo==:L ? (j:n) : (1:j))
                    AP[k] = A[i,j]
                    k += 1
                end
            end
            return AP
        end

        if elty in ["Float32", "Float64"]
            # pack matrices
            sAPU = pack(sA, :U)
            dsAPU = CuVector(sAPU)
            sAPL = pack(sA, :L)
            dsAPL = CuVector(sAPL)

            @testset "spmv!" begin
                alpha = rand(elty)
                beta = rand(elty)
                # generate vectors
                y = rand(elty,m)
                # copy to device
                dy = CuArray(y)
                # execute on host
                BLAS.spmv!('U',alpha,sAPU,x,beta,y)
                # execute on device
                CUBLAS.spmv!('U',alpha,dsAPU,dx,beta,dy)
                # compare results
                hy = Array(dy)
                @test y ≈ hy
                # execute on host
                BLAS.spmv!('U',alpha,sAPL,x,beta,y)
                # execute on device
                CUBLAS.spmv!('U',alpha,dsAPL,dx,beta,dy)
                # compare results
                hy = Array(dy)
                @test y ≈ hy
            end

            @testset "spr!" begin
                alpha = rand(elty)
                beta = rand(elty)
                # execute on host
                BLAS.spr!('U',alpha,x,sAPU)
                # execute on device
                CUBLAS.spr!('U',alpha,dx,dsAPU)
                # compare results
                hsAPU = Array(dsAPU)
                @test sAPU ≈ hsAPU
                # execute on host
                BLAS.spr!('U',alpha,x,sAPL)
                # execute on device
                CUBLAS.spr!('U',alpha,dx,dsAPL)
                # compare results
                hAPL = Array(dAPL)
                @test sAPL ≈ hAPL
            end
        end

        @testset "symv!" begin
            alpha = rand(elty)
            beta = rand(elty)
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
                alpha = rand(elty)
                beta = rand(elty)
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
            @test_throws DimensionMismatch CUBLAS.trmv!('U','N','N',dA,CUDA.rand(elty,m+1))
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
            @test_throws DimensionMismatch CUBLAS.trsv!('U','N','N',dA,CUDA.rand(elty,m+1))
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
        @testset "ldiv! of triangular types" begin
            @testset "ldiv!(::UpperTriangular)" begin
                A = rand(elty,m,m)
                A = A + transpose(A)
                dA = CuArray(A)
                x = rand(elty,m)
                dx = CuArray(x)
                dy = copy(dx)
                CUBLAS.ldiv!(UpperTriangular(dA), dy)
                y = UpperTriangular(A) \ x
                @test y ≈ Array(dy)
            end
            @testset "ldiv!(::UpperTriangular{Adjoint})" begin
                A = rand(elty,m,m)
                A = A + transpose(A)
                dA = CuArray(A)
                x = rand(elty,m)
                dx = CuArray(x)
                dy = copy(dx)
                CUBLAS.ldiv!(adjoint(UpperTriangular(dA)), dy)
                y = adjoint(UpperTriangular(A)) \ x
                @test y ≈ Array(dy)
            end
            @testset "ldiv!(::UpperTriangular{Transpose})" begin
                A = rand(elty,m,m)
                A = A + transpose(A)
                dA = CuArray(A)
                x = rand(elty,m)
                dx = CuArray(x)
                dy = copy(dx)
                CUBLAS.ldiv!(transpose(UpperTriangular(dA)), dy)
                y = transpose(UpperTriangular(A)) \ x
                @test y ≈ Array(dy)
            end
            @testset "ldiv!(::LowerTriangular)" begin
                A = rand(elty,m,m)
                A = A + transpose(A)
                dA = CuArray(A)
                x = rand(elty,m)
                dx = CuArray(x)
                dy = copy(dx)
                CUBLAS.ldiv!(LowerTriangular(dA), dy)
                y = LowerTriangular(A) \ x
                @test y ≈ Array(dy)
            end
            @testset "ldiv!(::LowerTriangular{Adjoint})" begin
                A = rand(elty,m,m)
                A = A + transpose(A)
                dA = CuArray(A)
                x = rand(elty,m)
                dx = CuArray(x)
                dy = copy(dx)
                CUBLAS.ldiv!(adjoint(LowerTriangular(dA)), dy)
                y = adjoint(LowerTriangular(A)) \ x
                @test y ≈ Array(dy)
            end
            @testset "ldiv!(::LowerTriangular{Transpose})" begin
                A = rand(elty,m,m)
                A = A + transpose(A)
                dA = CuArray(A)
                x = rand(elty,m)
                dx = CuArray(x)
                dy = copy(dx)
                CUBLAS.ldiv!(transpose(LowerTriangular(dA)), dy)
                y = transpose(LowerTriangular(A)) \ x
                @test y ≈ Array(dy)
            end
        end
        @testset "inv($TR)" for TR in (UpperTriangular, LowerTriangular, UnitUpperTriangular, UnitLowerTriangular)
            @test testf(x -> inv(TR(x)), rand(elty, m, m))
        end

        @testset "ger!" begin
            alpha = rand(elty)
            A = rand(elty,m,m)
            x = rand(elty,m)
            y = rand(elty,m)
            dA = CuArray(A)
            dx = CuArray(x)
            dy = CuArray(y)
            # perform rank one update
            dB = copy(dA)
            CUBLAS.ger!(alpha,dx,dy,dB)
            B = (alpha*x)*y' + A
            # move to host and compare
            hB = Array(dB)
            @test B ≈ hB
        end

        @testset "syr!" begin
            alpha = rand(elty)
            sA = rand(elty,m,m)
            sA = sA + transpose(sA)
            x = rand(elty,m)
            dx = CuArray(x)
            dB = CuArray(sA)
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
                alpha = rand(elty)
                hA = rand(elty,m,m)
                hA = hA + adjoint(hA)
                dB = CuArray(hA)
                x = rand(elty,m)
                dx = CuArray(x)
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
                alpha = rand(elty)
                hA = rand(elty,m,m)
                hA = hA + adjoint(hA)
                dB = CuArray(hA)
                x = rand(elty,m)
                dx = CuArray(x)
                y = rand(elty,m)
                dy = CuArray(y)
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

    @testset "StaticArray eltype" begin
       A = CuArray(rand(SVector{2, Float64}, 3, 3))
       B = CuArray(rand(Float64, 3, 1))
       C = A * B
       hC = Array(A) * Array(B)
       @test Array(C) ≈ hC
    end
end
