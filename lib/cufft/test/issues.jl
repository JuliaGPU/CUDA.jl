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
    @test size(fft(CUDA.rand(ComplexF32, 3, 5, 7, 11, 2), (2, 4))) == (3,5,7,11,2)
end

@testset "CUDA.jl_PR_Bug 2: check rfft with external batch stride > 1" begin
    # see https://github.com/JuliaGPU/CUDA.jl/pull/3052#issuecomment-4213439988
    x = CUDA.rand(Float32, 5, 3, 7, 4); xh = Array(x);
    xc = copy(x)
    y = rfft(x, (1, 3));
    @test x == xc
    @test maximum(abs.(Array(y) .- rfft(xh, (1, 3)))) < 1e-5
end

@testset "CUDA.jl_PR_Bug 3: ArgumentError for replicate directions" begin
    # see https://github.com/JuliaGPU/CUDA.jl/pull/3052#issuecomment-4213439988
    @test_throws ArgumentError plan_fft!(CUDA.rand(ComplexF32, 4, 4, 4), (1, 2, 1)) 
end
