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
            # testing without pivoting quickly results in inaccuracies along the diagonal
            # test with pivoting
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
            # generate bad matrices
            A_bad = vcat([rand(elty,m,m) for i in 1:9], [rand(elty, m, m+1)])
            # move to device
            d_A_bad = CuArray{elty, 2}[]
            for i in 1:length(A)
                push!(d_A_bad,CuArray(A_bad[i]))
            end
            @test_throws DimensionMismatch CUBLAS.getrf_batched!(d_A_bad, true)
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
            # testing without pivoting quickly results in inaccuracies along the diagonal
            # test with pivoting
            pivot, info, d_B = CUBLAS.getrf_batched(d_A, true)
            h_info = Array(info)
            h_pivot = Array(pivot)
            for Bs in 1:length(d_B)
                C   = lu(A[Bs])
                h_A = Array(d_B[Bs])
                #reconstruct L,U
                dL = Matrix(one(elty)*I, m, m)
                dU = zeros(elty,(m,m))
                k = h_info[Bs]
                if( k >= 0 )
                    dL += tril(h_A,-k-1)
                    dU += triu(h_A,k)
                end
                #compare pivots
                @test length(setdiff(h_pivot[:,Bs],C.p)) == 0
                #make device pivot matrix
                P = Matrix(1.0*I, m, m)
                for row in 1:m
                    temp = copy(P[row,:])
                    P[row,:] = P[h_pivot[row,Bs],:]
                    P[h_pivot[row,Bs],:] = temp
                end
                @test inv(P)*dL*dU ≈ inv(C.P) * C.L * C.U
            end
        end

        @testset "getrf_strided_batched!" begin
            Random.seed!(1)
            local k
            # generate strided matrix
            A = rand(elty,m,m,10)
            # move to device
            d_A = CuArray(A)
            # testing without pivoting quickly results in inaccuracies along the diagonal
            # test with pivoting
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
            # generate bad strided matrix
            A = rand(elty,m,m+1,10)
            # move to device
            d_A = CuArray(A)
            @test_throws DimensionMismatch CUBLAS.getrf_strided_batched!(d_A, true)
        end

        @testset "getrf_strided_batched" begin
            local k
            # generate strided matrix
            A = rand(elty,m,m,10)
            # move to device
            d_A = CuArray(A)
            # testing without pivoting quickly results in inaccuracies along the diagonal
            # test with pivoting
            pivot, info, d_B = CUBLAS.getrf_strided_batched(d_A, true)
            h_info = Array(info)
            h_pivot = Array(pivot)
            for Bs in 1:size(d_B, 3)
                C   = lu(A[:,:,Bs])
                h_A = Array(d_B[:,:,Bs])
                #reconstruct L,U
                dL = Matrix(one(elty)*I, m, m)
                dU = zeros(elty,(m,m))
                k = h_info[Bs]
                if( k >= 0 )
                    dL += tril(h_A,-k-1)
                    dU += triu(h_A,k)
                end
                #compare pivots
                @test length(setdiff(h_pivot[:,Bs],C.p)) == 0
                #make device pivot matrix
                P = Matrix(1.0*I, m, m)
                for row in 1:m
                    temp = copy(P[row,:])
                    P[row,:] = P[h_pivot[row,Bs],:]
                    P[h_pivot[row,Bs],:] = temp
                end
                @test inv(P)*dL*dU ≈ inv(C.P) * C.L * C.U
            end
        end

        for (opchar,opfun) in (('N',identity), ('T',transpose), ('C',adjoint))

            @testset "getrs_batched!" begin
                A                   = [rand(elty,n,n) for _ in 1:k]
                d_A                 = [CuArray(a) for a in A]
                d_A2                = deepcopy(d_A)
                d_pivot, info, d_LU = CUDA.CUBLAS.getrf_batched!(d_A, true)
                @test d_LU == d_A
                d_pivot2            = similar(d_pivot)
                info2               = similar(info)
                CUDA.CUBLAS.getrf_batched!(d_A2, d_pivot2, info2)
                @test isapprox(d_pivot, d_pivot2)
                @test isapprox(info, info2)
                B                   = [rand(elty,n,m) for _ in 1:k]
                d_B                 = [CuArray(b) for b in B]
                info2, d_Bhat       = CUDA.CUBLAS.getrs_batched!(opchar, d_LU, d_B, d_pivot)
                @test d_Bhat == d_B
                h_Bhat              = [collect(bh) for bh in d_Bhat]
                for i in 1:k
                    @test h_Bhat[i] ≈ opfun(A[i]) \ B[i]
                end
                
                # generate bad matrices
                A_bad = vcat([rand(elty,m,m) for i in 1:9], [rand(elty, m, m+1)])
                # move to device
                d_A_bad = CuArray{elty, 2}[]
                for i in 1:length(A_bad)
                    push!(d_A_bad,CuArray(A_bad[i]))
                end
                @test_throws DimensionMismatch CUBLAS.getrs_batched!(opchar, d_A_bad, d_B, d_pivot)
                # generate bad matrices
                A_bad = [rand(elty,m+1,m+1) for i in 1:10]
                # move to device
                d_A_bad = CuArray{elty, 2}[]
                for i in 1:length(A_bad)
                    push!(d_A_bad,CuArray(A_bad[i]))
                end
                @test_throws DimensionMismatch CUBLAS.getrs_batched!(opchar, d_A_bad, d_B, d_pivot)
            end

            @testset "getrs_batched" begin
                A                   = [rand(elty,n,n) for _ in 1:k];
                d_A                 = [CuArray(a) for a in A];
                d_A2                = deepcopy(d_A);
                d_pivot, info, d_LU = CUDA.CUBLAS.getrf_batched(d_A, true);
                @test d_LU != d_A
                d_pivot2            = similar(d_pivot);
                info2               = similar(info);
                CUDA.CUBLAS.getrf_batched(d_A2, d_pivot2, info2);
                @test isapprox(d_pivot, d_pivot2)
                @test isapprox(info, info2)
                B                   = [rand(elty,n,m) for _ in 1:k];
                d_B                 = [CuArray(b) for b in B];
                info2, d_Bhat       = CUDA.CUBLAS.getrs_batched(opchar, d_LU, d_B, d_pivot);
                @test d_Bhat != d_B
                h_Bhat              = [collect(bh) for bh in d_Bhat];
                for i in 1:k
                    @test h_Bhat[i] ≈ opfun(A[i]) \ B[i]
                end
            end

            @testset "getrs_strided_batched!" begin
                A                   = rand(elty,n,n,k)
                d_A                 = CuArray(A)
                d_A2                = copy(d_A)
                d_pivot, info, d_LU = CUDA.CUBLAS.getrf_strided_batched!(d_A, true)
                @test d_LU == d_A
                d_pivot2            = similar(d_pivot)
                info2               = similar(info)
                CUDA.CUBLAS.getrf_strided_batched!(d_A2, d_pivot2, info2)
                @test isapprox(d_pivot, d_pivot2)
                @test isapprox(info, info2)
                B                   = rand(elty,n,m,k)
                d_B                 = CuArray(B)
                info2, d_Bhat       = CUDA.CUBLAS.getrs_strided_batched!(opchar, d_LU, d_B, d_pivot)
                @test d_Bhat == d_B
                h_Bhat              = collect(d_Bhat)
                for i in 1:k
                    @test h_Bhat[:,:,i] ≈ opfun(A[:,:,i]) \ B[:,:,i]
                end

                A_bad               = rand(elty,n+1,n,k)
                d_A_bad             = CuArray(A_bad)
                @test_throws DimensionMismatch CUDA.CUBLAS.getrs_strided_batched!(opchar, d_A_bad, d_B, d_pivot)
                A_bad               = rand(elty,n+1,n+1,k)
                d_A_bad             = CuArray(A_bad)
                @test_throws DimensionMismatch CUDA.CUBLAS.getrs_strided_batched!(opchar, d_A_bad, d_B, d_pivot)
            end

            @testset "getrs_strided_batched" begin
                A                   = rand(elty,n,n,k);
                d_A                 = CuArray(A);
                d_A2                = copy(d_A);
                d_pivot, info, d_LU = CUDA.CUBLAS.getrf_strided_batched(d_A, true);
                @test d_LU != d_A
                d_pivot2            = similar(d_pivot);
                info2               = similar(info);
                CUDA.CUBLAS.getrf_strided_batched(d_A2, d_pivot2, info2);
                @test isapprox(d_pivot, d_pivot2)
                @test isapprox(info, info2)
                B                   = rand(elty,n,m,k);
                d_B                 = CuArray(B);
                info2, d_Bhat       = CUDA.CUBLAS.getrs_strided_batched(opchar, d_LU, d_B, d_pivot);
                @test d_Bhat != d_B
                h_Bhat              = collect(d_Bhat);
                for i in 1:k
                    @test h_Bhat[:,:,i] ≈ opfun(A[:,:,i]) \ B[:,:,i]
                end
            end

        end

        @testset "getri_strided_batched" begin
            # generate strided matrix
            A = rand(elty,m,m,10)
            # move to device
            d_A = CuArray(A)
            d_B = similar(d_A)
            pivot, info = CUBLAS.getrf_strided_batched!(d_A, true)
            info = CUBLAS.getri_strided_batched!(d_A, d_B, pivot)
            h_info = Array(info)
            for Cs in 1:size(d_A,3)
                B   = inv(A[:,:,Cs])
                @test h_info[Cs] == 0
                @test B ≈ Array(d_B[:,:,Cs]) rtol=1e-3
            end

            A_bad = rand(elty,m+1,m,10)
            d_A_bad = CuArray(A_bad)
            d_B = similar(d_A)
            pivot, info = CUBLAS.getrf_strided_batched!(d_A, true)
            @test_throws DimensionMismatch CUBLAS.getri_strided_batched!(d_A_bad, d_B, pivot)
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

            d_A = CuArray{elty, 2}[]
            for i in 1:length(A)
                push!(d_A,CuArray(A[i]))
            end
            pivot, info = CUBLAS.getrf_batched!(d_A, true)
            h_info = Array(info)
            for Cs in 1:length(h_info)
                @test h_info[Cs] == 0
            end
            d_C = CuMatrix{elty}[similar(d_A[1]) for i in 1:length(d_A)]
            info = CUBLAS.getri_batched!(d_A, d_C, pivot)
            h_info = Array(info)
            for Cs in 1:length(d_C)
                C   = inv(A[Cs])
                h_C = Array(d_C[Cs])
                @test h_info[Cs] == 0
                @test C ≈ h_C rtol=1e-2
            end

            A_bad = [rand(elty,m+1,m) for i in 1:10]
            d_A_bad = CuArray{elty, 2}[]
            for i in 1:length(A)
                push!(d_A_bad,CuArray(A_bad[i]))
            end
            @test_throws DimensionMismatch CUBLAS.getri_batched(d_A_bad, pivot)
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

            # matinv_batched only supports matrices smaller than 32x32
            A = [rand(elty,64,64) for i in 1:10]
            # move to device
            d_A_too_big = CuArray{elty, 2}[]
            for i in 1:length(A)
                push!(d_A_too_big,CuArray(A[i]))
            end
            @test_throws ArgumentError("matinv requires all matrices be smaller than 32 x 32") CUBLAS.matinv_batched(d_A_too_big)
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
            tau, d_B = CUBLAS.geqrf_batched(d_A)
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

            # generate bad matrices
            A = [rand(elty,n,k) for i in 1:10]
            C = [rand(elty,n+1,k) for i in 1:10]
            # move to device
            d_A = CuArray{elty, 2}[]
            d_C = CuArray{elty, 2}[]
            for i in 1:length(A)
                push!(d_A,CuArray(A[i]))
                push!(d_C,CuArray(C[i]))
            end
            @test_throws DimensionMismatch CUBLAS.gels_batched!('N',d_A, d_C)
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
        y = rand(elty,n)
        # move to device
        d_A = CuArray(A)
        d_C = CuArray(C)
        d_x = CuArray(x)
        d_y = CuArray(y)
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
        @testset "diagm" begin
            d_fX = LinearAlgebra.diagm(d_x)
            @test eltype(d_fX) == eltype(d_x)
        end
        @testset "diagonal -- mul!, rmul!, lmul!" begin
            XA = rand(elty,m,n)
            d_XA = CuArray(XA)
            d_X = Diagonal(d_x)
            mul!(d_XA, d_X, d_A)
            @test Array(d_XA) ≈ Diagonal(x) * A
            
            XA = rand(elty,m,n)
            d_XA = CuArray(XA)
            d_X = Diagonal(d_x)
            lmul!(d_X, d_XA)
            @test Array(d_XA) ≈ Diagonal(x) * XA

            AY = rand(elty,m,n)
            d_AY = CuArray(AY)
            d_Y = Diagonal(d_y)
            mul!(d_AY, d_A, d_Y)
            @test Array(d_AY) ≈ A * Diagonal(y)
            
            AY = rand(elty,m,n)
            d_AY = CuArray(AY)
            d_Y = Diagonal(d_y)
            rmul!(d_AY, d_Y)
            @test Array(d_AY) ≈ AY * Diagonal(y)

            YA = rand(elty,n,m)
            d_YA = CuArray(YA)
            d_Y = Diagonal(d_y)
            mul!(d_YA, d_Y, transpose(d_A))
            @test Array(d_YA) ≈ Diagonal(y) * transpose(A)

            AX = rand(elty,n,m)
            d_AX = CuArray(AX)
            d_X = Diagonal(d_x)
            mul!(d_AX, transpose(d_A), d_X)
            @test Array(d_AX) ≈ transpose(A) * Diagonal(x)

            YA = rand(elty,n,m)
            d_YA = CuArray(YA)
            d_Y = Diagonal(d_y)
            mul!(d_YA, d_Y, d_A')
            @test Array(d_YA) ≈ Diagonal(y) * A'

            AX = rand(elty,n,m)
            d_AX = CuArray(AX)
            d_X = Diagonal(d_x)
            mul!(d_AX, d_A', d_X)
            @test Array(d_AX) ≈ A' * Diagonal(x)

            @test Array(d_X) == Diagonal(Array(d_x))
        end
    end # extensions

    @testset "StaticArray eltype" begin
       A = CuArray(rand(SVector{2, Float32}, 3, 3))
       B = CuArray(rand(Float32, 3, 3))
       C = A * B
       hC = Array(A) * Array(B)
       @test Array(C) ≈ hC
    end
end # elty

@testset "rmul/lmul with mixed eltypes ($Tr, $Tc)" for (Tr, Tc) in ((Float32, ComplexF32), (Float64, ComplexF64))
    x    = rand(Tr,m)
    d_x  = CuArray(x)
    XA   = rand(Tc,m,n)
    d_XA = CuArray(XA)
    d_X  = Diagonal(d_x)
    lmul!(d_X, d_XA)
    @test Array(d_XA) ≈ Diagonal(x) * XA

    x    = rand(Tr,m)
    d_x  = CuArray(x)
    XA   = rand(Tc,n,m)
    d_AX = transpose(CuArray(XA))
    d_X  = Diagonal(d_x)
    lmul!(d_X, d_AX)
    @test Array(d_AX) ≈ Diagonal(x) * transpose(XA)

    x    = rand(Tr,m)
    d_x  = CuArray(x)
    XA   = rand(Tc,n,m)
    d_AX = adjoint(CuArray(XA))
    d_X  = Diagonal(d_x)
    lmul!(d_X, d_AX)
    @test Array(d_AX) ≈ Diagonal(x) * adjoint(XA)

    y    = rand(Tr,n)
    d_y  = CuArray(y)
    AY   = rand(Tc,m,n)
    d_AY = CuArray(AY)
    d_Y  = Diagonal(d_y)
    rmul!(d_AY, d_Y)
    @test Array(d_AY) ≈ AY * Diagonal(y)

    y    = rand(Tr,n)
    d_y  = CuArray(y)
    AY   = rand(Tc,n,m)
    d_YA = transpose(CuArray(AY))
    d_Y  = Diagonal(d_y)
    d_YA = rmul!(d_YA, d_Y)
    @test Array(d_YA) ≈ transpose(AY) * Diagonal(y)

    y    = rand(Tr,n)
    d_y  = CuArray(y)
    AY   = rand(Tc,n,m)
    d_YA = adjoint(CuArray(AY))
    d_Y  = Diagonal(d_y)
    d_YA = rmul!(d_YA, d_Y)
    @test Array(d_YA) ≈ adjoint(AY) * Diagonal(y)
end
