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
    @testset for elty in [Float32, Float64, ComplexF32, ComplexF64]

        @testset "mul! C = $f(A) *  $g(B) * $Ts(a) + C * $Ts(b)" for f in (identity, transpose, adjoint), g in (identity, transpose, adjoint), Ts in (Int, elty)
            C, A, B = rand(elty, 5, 5), rand(elty, 5, 5), rand(elty, 5, 5)
            dC, dA, dB = CuArray(C), CuArray(A), CuArray(B)
            mul!(dC, f(dA), g(dB), Ts(1), Ts(2))
            mul!(C, f(A), g(B), Ts(1), Ts(2))
            @test Array(dC) ≈ C
        end

        @testset "hermitian" begin
            C, A, B = rand(elty, 5, 5), Hermitian(rand(elty, 5, 5)), rand(elty, 5, 5)
            dC, dA, dB = CuArray(C), Hermitian(CuArray(A)), CuArray(B)
            mul!(dC, dA, dB)
            mul!(C, A, B)
            @test Array(dC) ≈ C
        end

        @testset "gemm!" begin
            alpha = rand(elty)
            beta = rand(elty)
            A = rand(elty,m,k)
            B = rand(elty,k,n)
            C1 = rand(elty,m,n)
            C2 = copy(C1)
            d_A = CuArray(A)
            d_B = CuArray(B)
            d_C1 = CuArray(C1)
            d_C2 = CuArray(C2)
            hA = rand(elty,m,m)
            hA = hA + hA'
            dhA = CuArray(hA)
            sA = rand(elty,m,m)
            sA = sA + transpose(sA)
            dsA = CuArray(sA)

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
            @test_throws DimensionMismatch CUBLAS.gemm!('N','N',one(elty),d_A,dsA,one(elty),d_C1)
            @test_throws DimensionMismatch mul!(d_C1, d_A, dsA)
        end
        @testset "strided gemm!" begin
            denseA = CUDA.rand(elty, 4,4)
            denseB = CUDA.rand(elty, 4,4)
            denseC = CUDA.zeros(elty, 4,4)

            stridedA = view(denseA, 1:2, 1:2)::SubArray
            stridedB = view(denseB, 1:2, 1:2)::SubArray
            stridedC = view(denseC, 1:2, 1:2)::SubArray

            CUBLAS.gemm!('N', 'N', true, stridedA, stridedB, false, stridedC)
            @test Array(stridedC) ≈ Array(stridedA) * Array(stridedB)

            stridedC .= 0
            mul!(stridedC, stridedA, stridedB, true, false)
            @test Array(stridedC) ≈ Array(stridedA) * Array(stridedB)
        end
        if capability(device()) > v"5.0"
            @testset "gemmEx!" begin
                A = rand(elty,m,k)
                B = rand(elty,k,n)
                C1 = rand(elty,m,n)
                d_A = CuArray(A)
                d_B = CuArray(B)
                d_C1 = CuArray(C1)
                α = rand(elty)
                β = rand(elty)
                CUBLAS.gemmEx!('N','N',α,d_A,d_B,β,d_C1)
                h_C1 = Array(d_C1)
                C1 = (α*A)*B + β*C1
                # compare
                @test C1 ≈ h_C1
                d_Cbad = CUDA.zeros(elty, m+1, n-1) 
                @test_throws DimensionMismatch CUBLAS.gemmEx!('N','N',α,d_A,d_B,β,d_Cbad)
            end
        end
        @testset "gemm" begin
            A = rand(elty,m,k)
            B = rand(elty,k,n)
            d_A = CuArray(A)
            d_B = CuArray(B)
            d_C1 = CUBLAS.gemm('N','N',d_A,d_B)
            C1 = A*B
            C2 = d_A * d_B
            # compare
            h_C1 = Array(d_C1)
            h_C2 = Array(C2)
            @test C1 ≈ h_C1
            @test C1 ≈ h_C2
        end
        @testset "symm!" begin
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
            d_Bbad = CuArray(Bbad)
            CUBLAS.symm!('L','U',alpha,dsA,d_B,beta,d_C)
            C = (alpha*sA)*B + beta*C
            # compare
            h_C = Array(d_C)
            @test C ≈ h_C
            @test_throws DimensionMismatch CUBLAS.symm!('L','U',alpha,dsA,d_Bbad,beta,d_C)
        end

        @testset "symm" begin
            sA = rand(elty,m,m)
            sA = sA + transpose(sA)
            dsA = CuArray(sA)
            B = rand(elty,m,n)
            C = rand(elty,m,n)
            Bbad = rand(elty,m+1,n+1)
            d_B = CuArray(B)
            d_C = CuArray(C)
            d_Bbad = CuArray(Bbad)
            d_C = CUBLAS.symm('L','U',dsA,d_B)
            C = sA*B
            # compare
            h_C = Array(d_C)
            @test C ≈ h_C
            @test_throws DimensionMismatch CUBLAS.symm('L','U',dsA,d_Bbad)
        end

        if elty <: Complex
            @testset "hemm!" begin
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
                CUBLAS.hemm!('L','L',alpha,dhA,d_B,beta,d_C)
                # move to host and compare
                h_C = Array(d_C)
                @test C ≈ h_C
            end
            @testset "hemm" begin
                hA = rand(elty,m,m)
                hA = hA + hA'
                dhA = CuArray(hA)
                B = rand(elty,m,n)
                d_B = CuArray(B)
                C = hA*B
                d_C = CUBLAS.hemm('L','U',dhA,d_B)
                # move to host and compare
                h_C = Array(d_C)
                @test C ≈ h_C
            end
        end
    end
    @testset "elty = $elty" for elty in [Float16, Float32, Float64, ComplexF32, ComplexF64]
        elty == Float16 && capability(device()) < v"5.3" && continue

        alpha = rand(elty)
        beta = rand(elty)
        # generate matrices
        bA = [rand(elty,m,k) for i in 1:10]
        bB = [rand(elty,k,n) for i in 1:10]
        bC = [rand(elty,m,n) for i in 1:10]
        # move to device
        bd_A = CuArray{elty, 2}[]
        bd_B = CuArray{elty, 2}[]
        bd_C = CuArray{elty, 2}[]
        bd_A_bad = CuArray{elty, 2}[]
        bd_bad = CuArray{elty, 2}[]
        for i in 1:length(bA)
            push!(bd_A,CuArray(bA[i]))
            push!(bd_B,CuArray(bB[i]))
            push!(bd_C,CuArray(bC[i]))
            if i < length(bA) - 2
                push!(bd_bad,CuArray(bC[i]))
                push!(bd_A_bad,CuArray(bA[i]))
            else
                push!(bd_A_bad,CUDA.rand(elty, m+1, k-1))
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
            @test_throws DimensionMismatch CUBLAS.gemm_batched!('N','N',alpha,bd_A_bad,bd_B,beta,bd_C)
        end

        @testset "gemm_batched" begin
            bd_C = CUBLAS.gemm_batched('N','N',bd_A,bd_B)
            for i in 1:length(bA)
                bC[i] = bA[i]*bB[i]
                h_C = Array(bd_C[i])
                @test bC[i] ≈ h_C
            end
            @test_throws DimensionMismatch CUBLAS.gemm_batched('N','N',alpha,bd_A,bd_bad)
            @test_throws DimensionMismatch CUBLAS.gemm_batched('N','N',alpha,bd_A_bad,bd_B)
        end

        @testset "gemmBatchedEx!" begin
            # C = (alpha*A)*B + beta*C
            CUBLAS.gemmBatchedEx!('N','N',alpha,bd_A,bd_B,beta,bd_C)
            for i in 1:length(bd_C)
                bC[i] = (alpha*bA[i])*bB[i] + beta*bC[i]
                h_C = Array(bd_C[i])
                #compare
                @test bC[i] ≈ h_C
            end
            @test_throws DimensionMismatch CUBLAS.gemmBatchedEx!('N','N',alpha,bd_A,bd_bad,beta,bd_C)
            @test_throws DimensionMismatch CUBLAS.gemmBatchedEx!('N','N',alpha,bd_A_bad,bd_B,beta,bd_C)
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

        @testset "gemmStridedBatchedEx!" begin
            CUBLAS.gemmStridedBatchedEx!('N', 'N', alpha, bd_A, bd_B, beta, bd_C)
            for i in 1:nbatch
                bC[:, :, i] = (alpha * bA[:, :, i]) * bB[:, :, i] + beta * bC[:, :, i]
            end
            h_C = Array(bd_C)
            @test bC ≈ h_C
            @test_throws DimensionMismatch CUBLAS.gemmStridedBatchedEx!('N', 'N', alpha, bd_A, bd_B, beta, bd_bad)
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
    end

    if CUDA.CUBLAS.version() >= v"12.4.2"
        @testset "elty = $elty" for elty in [Float32, Float64]
            num_groups = 10
            group_sizes = collect(1:num_groups)
            transA = ['N' for i in 1:num_groups]
            transB = ['N' for i in 1:num_groups]
            alpha = rand(elty, num_groups)
            beta = rand(elty, num_groups)
            # generate matrices
            bA = [[rand(elty,3*i,2*i) for j in 1:group_sizes[i]] for i in 1:num_groups]
            bB = [[rand(elty,2*i,5*i) for j in 1:group_sizes[i]] for i in 1:num_groups]
            bC = [[rand(elty,3*i,5*i) for j in 1:group_sizes[i]] for i in 1:num_groups]
            # move to device
            bd_A = [[CuArray(bA[i][j]) for j in 1:group_sizes[i]] for i in 1:num_groups]
            bd_B = [[CuArray(bB[i][j]) for j in 1:group_sizes[i]] for i in 1:num_groups]
            bd_C = [[CuArray(bC[i][j]) for j in 1:group_sizes[i]] for i in 1:num_groups]
            bd_A_bad1 = [[CuArray(bA[i][j]) for j in 1:group_sizes[i]] for i in 1:num_groups-1]
            bd_A_bad2 = [[CuArray(bA[i][j]) for j in 1:group_sizes[i]-1] for i in 1:num_groups]
            bd_A_bad3 = [[CUDA.rand(elty, 3*i+1,2*i - 1) for j in 1:group_sizes[i]] for i in 1:num_groups]
            @testset "gemm_grouped_batched!" begin
                # C = (alpha*A)*B + beta*C
                CUBLAS.gemm_grouped_batched!(transA,transB,alpha,bd_A,bd_B,beta,bd_C)
                for i in 1:num_groups, j in 1:group_sizes[i]
                    bC[i][j] = alpha[i] * bA[i][j] * bB[i][j] + beta[i] * bC[i][j]
                    h_C = Array(bd_C[i][j])
                    @test bC[i][j] ≈ h_C
                end
                @test_throws DimensionMismatch CUBLAS.gemm_grouped_batched!(transA,transB,alpha,bd_A_bad1,bd_B,beta,bd_C)
                @test_throws DimensionMismatch CUBLAS.gemm_grouped_batched!(transA,transB,alpha,bd_A_bad2,bd_B,beta,bd_C)
                @test_throws DimensionMismatch CUBLAS.gemm_grouped_batched!(transA,transB,alpha,bd_A_bad3,bd_B,beta,bd_C)
            end

            @testset "gemm_grouped_batched" begin
                bd_C = CUBLAS.gemm_grouped_batched(transA,transB,bd_A,bd_B)
                for i in 1:num_groups, j in 1:group_sizes[i]
                    bC[i][j] = bA[i][j] * bB[i][j]
                    h_C = Array(bd_C[i][j])
                    @test bC[i][j] ≈ h_C
                end
            end
        end
    end

    # Group size hardcoded to one
    if CUDA.CUBLAS.version() >= v"12.4.2"
        @testset "elty = $elty" for elty in [Float32, Float64]

            transA = ['N' for i in 1:10]
            transB = ['N' for i in 1:10]
            alpha = rand(elty, 10)
            beta = rand(elty, 10)
            # generate matrices
            bA = [rand(elty,3*i,2*i) for i in 1:10]
            bB = [rand(elty,2*i,5*i) for i in 1:10]
            bC = [rand(elty,3*i,5*i) for i in 1:10]
            # move to device
            bd_A = CuArray{elty, 2}[]
            bd_B = CuArray{elty, 2}[]
            bd_C = CuArray{elty, 2}[]
            bd_A_bad1 = CuArray{elty, 2}[]
            bd_A_bad2 = CuArray{elty, 2}[]
            for i in 1:length(bA)
                push!(bd_A,CuArray(bA[i]))
                push!(bd_B,CuArray(bB[i]))
                push!(bd_C,CuArray(bC[i]))
                if i < length(bA) - 1
                    push!(bd_A_bad1,CuArray(bA[i]))
                    push!(bd_A_bad2,CuArray(bA[i]))
                else
                    push!(bd_A_bad2,CUDA.rand(elty, 3*i+1, 2*i-1))
                end
            end

            @testset "gemm_grouped_batched!" begin
                # C = (alpha*A)*B + beta*C
                CUBLAS.gemm_grouped_batched!(transA,transB,alpha,bd_A,bd_B,beta,bd_C)
                for i in 1:length(bd_C)
                    bC[i] = alpha[i] * bA[i] * bB[i] + beta[i] * bC[i]
                    h_C = Array(bd_C[i])
                    @test bC[i] ≈ h_C
                end
                @test_throws DimensionMismatch CUBLAS.gemm_grouped_batched!(transA,transB,alpha,bd_A_bad1,bd_B,beta,bd_C)
                @test_throws DimensionMismatch CUBLAS.gemm_grouped_batched!(transA,transB,alpha,bd_A_bad2,bd_B,beta,bd_C)
            end

            @testset "gemm_grouped_batched" begin
                bd_C = CUBLAS.gemm_grouped_batched(transA,transB,bd_A,bd_B)
                for i in 1:length(bd_C)
                    bC[i] = bA[i] * bB[i]
                    h_C = Array(bd_C[i])
                    @test bC[i] ≈ h_C
                end
            end
        end
    end

    starting_mode = CUDA.math_mode()
    starting_precision = CUDA.math_precision()
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
        try
            # test in fast math mode too
            for precision in (:Float16, :BFloat16, :TensorFloat32), (AT, CT) in ((Float32, Float32), (ComplexF32, ComplexF32)) 
                CUDA.math_mode!(CUDA.FAST_MATH; precision=precision)
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
            CUDA.math_mode!(CUDA.FAST_MATH; precision = :Bad)
            @test_throws ArgumentError("Unknown reduced precision type Bad") CUBLAS.gemmExComputeType(Float32, Float32, Float32, m, k, n)
        finally
            CUDA.math_mode!(starting_mode; precision = starting_precision)
        end

        # also test an unsupported combination (falling back to GPUArrays)
        if VERSION < v"1.11-"   # JuliaGPU/CUDA.jl#2441
            AT=BFloat16
            BT=Int32
            CT=Float64

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
