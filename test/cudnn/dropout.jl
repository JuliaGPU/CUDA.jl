using Test, CUDA, Statistics
using CUDA.CUDNN: 
    cudnnDropoutForward,
    cudnnDropoutForward!,
    cudnnDropoutBackward,
    cudnnDropoutSeed,
    cudnnDropoutDescriptor,
        cudnnDropoutDescriptor_t,
        cudnnCreateDropoutDescriptor,
        cudnnSetDropoutDescriptor,
        cudnnGetDropoutDescriptor,
        cudnnRestoreDropoutDescriptor,
        cudnnDestroyDropoutDescriptor,
    cudnnDropoutGetStatesSize,
    cudnnDropoutGetReserveSpaceSize,
    handle

@testset "cudnn/dropout" begin

    @test cudnnDropoutDescriptor(C_NULL) isa cudnnDropoutDescriptor
    @test Base.unsafe_convert(Ptr, cudnnDropoutDescriptor(C_NULL)) isa Ptr
    @test cudnnDropoutDescriptor(0.5) isa cudnnDropoutDescriptor

    N,P = 1000, 0.7
    x = CUDA.rand(N)
    d = cudnnDropoutDescriptor(P)
    cudnnDropoutSeed[] = 1
    y = cudnnDropoutForward(x; dropout = P)
    @test isapprox(mean(Array(y).==0), P; atol = 3/sqrt(N))
    @test y == cudnnDropoutForward(x, d)
    @test y == cudnnDropoutForward!(similar(x), x; dropout = P)
    @test y == cudnnDropoutForward!(similar(x), x, d)
    cudnnDropoutSeed[] = -1

end
