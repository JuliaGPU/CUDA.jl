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
    x = CUDA.zeros(ComplexF32, 4)
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
    x = CUDA.zeros(Float32, 4)
    p = plan_rfft(x)
    pinv = inv(p)
    @test p isa AbstractFFTs.Plan{Float32}
    @test eltype(p) === Float32
    @test pinv isa AbstractFFTs.Plan{ComplexF32}
    @test eltype(pinv) === ComplexF32
end
