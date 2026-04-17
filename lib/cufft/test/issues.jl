@testset "CUDA.jl#1268" begin
    N=2^20
    v0 = CuArray(ones(N)+im*ones(N))

    v = CuArray(ones(N)+im*ones(N))
    plan = cuFFT.plan_fft!(v,1)
    @test fetch(
        Threads.@spawn begin
            inv(plan)*(plan*v)
            isapprox(v,v0)
        end
    )
end

@testset "CUDA.jl#1311" begin
    x = ones(8, 9)
    p = plan_rfft(x)
    y = similar(p * x)
    mul!(y, p, x)

    dx = CuArray(x)
    dp = plan_rfft(dx)
    dy = similar(dp * dx)
    mul!(dy, dp, dx)

    @test Array(dy) ≈ y
end

@testset "CUDA.jl#2409" begin
    x = CUDACore.zeros(ComplexF32, 4)
    p = plan_ifft(x)
    @test p isa AbstractFFTs.ScaledPlan
    # Initialize sz ref to invalid value
    sz = Ref{Csize_t}(typemax(Csize_t))
    # This will call the new convert method for ScaledPlan
    cuFFT.cufftGetSize(p, sz)
    # Make sure the value was modified
    @test sz[] != typemax(Csize_t)
end

@testset "CUDA.jl#2504" begin
    x = CUDACore.zeros(Float32, 4)
    p = plan_rfft(x)
    pinv = inv(p)
    @test p isa AbstractFFTs.Plan{Float32}
    @test eltype(p) === Float32
    @test pinv isa AbstractFFTs.Plan{ComplexF32}
    @test eltype(pinv) === ComplexF32
end

@testset "CUDA.jl_PR_Bug: 1 get_batch_dims for 5D+" begin
    # see https://github.com/JuliaGPU/CUDA.jl/pull/3052#issuecomment-4213439988
    @test size(fft(CUDACore.rand(ComplexF32, 3, 5, 7, 11, 2), (2, 4))) == (3,5,7,11,2)
end

@testset "CUDA.jl_PR_Bug 2: check rfft with external batch stride > 1" begin
    # see https://github.com/JuliaGPU/CUDA.jl/pull/3052#issuecomment-4213439988
    x = CUDACore.rand(Float32, 5, 3, 7, 4); xh = Array(x);
    xc = copy(x)
    y = rfft(x, (1, 3));
    @test x == xc
    @test maximum(abs.(Array(y) .- rfft(xh, (1, 3)))) < 1e-5
end

@testset "CUDA.jl_PR_Bug 3: ArgumentError for replicate directions" begin
    # see https://github.com/JuliaGPU/CUDA.jl/pull/3052#issuecomment-4213439988
    X = CUDACore.rand(ComplexF32, 4, 4, 4)
    @test_throws ArgumentError plan_fft!(X, (1, 2, 1))
    @test_throws ArgumentError plan_fft!(X, [1, 2, 1])
end

@testset "PR3052: inembed[0] >= n[0] when internal batch dim is 1" begin
    # region=(2,3) in 3D forces internal_batch_dims=(1,) and idist=1, which used
    # to violate cuFFT's documented inembed[0] >= n[0] precondition.
    x = CUDACore.rand(ComplexF32, 4, 5, 7); xh = Array(x)
    y = fft(x, (2, 3))
    @test maximum(abs.(Array(y) .- fft(xh, (2, 3)))) < 1e-4
end

@testset "PR3052: scratch-buffer copy preserves input across alignment hazard" begin
    # Float32 rfft with odd dim sizes between/before transform dims forces
    # multiple external batches at misaligned linear offsets. Verify that
    # (a) the input array is not mutated, and (b) the result matches CPU FFTW.
    for region in [(1, 3), (1, 4), (3, 4)]
        x = CUDACore.rand(Float32, 5, 3, 7, 9)
        xref = copy(x)
        y = rfft(x, region)
        @test x == xref                                          # input preserved
        @test maximum(abs.(Array(y) .- rfft(Array(xref), region))) < 1e-4
    end
end

@testset "PR3052: rfft → irfft round-trip exercises C2R external-batch path" begin
    x = CUDACore.rand(Float32, 5, 3, 7, 9)
    xref = copy(x)
    y = rfft(x, (1, 3))
    @test x == xref
    z = irfft(y, size(x, 1), (1, 3))
    @test maximum(abs.(Array(z) .- Array(xref))) < 1e-3
end
