using Statistics
using CUDNN:
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
    cudnnDropoutGetReserveSpaceSize

@test cudnnDropoutDescriptor(C_NULL) isa cudnnDropoutDescriptor
@test Base.unsafe_convert(Ptr, cudnnDropoutDescriptor(C_NULL)) isa Ptr
@test cudnnDropoutDescriptor(0.5) isa cudnnDropoutDescriptor

N,P = 1000, 0.7
x = CUDA.rand(N)
d = cudnnDropoutDescriptor(P)
cudnnDropoutSeed[] = 1  # only for testing; this makes dropout deterministic but slow
y = cudnnDropoutForward(x; dropout = P) |> Array
@test isapprox(mean(y.==0), P; atol = 3/sqrt(N))
@test y == cudnnDropoutForward(x, d) |> Array
@test y == cudnnDropoutForward!(similar(x), x; dropout = P) |> Array
@test y == cudnnDropoutForward!(similar(x), x, d) |> Array
cudnnDropoutSeed[] = -1
