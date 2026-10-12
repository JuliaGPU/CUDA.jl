using Statistics
using cuDNN:
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
x = CuArray(rand(Float32, N))
d = cudnnDropoutDescriptor(P)
cudnnDropoutSeed[] = 1  # only for testing; this makes dropout deterministic but slow
y = cudnnDropoutForward(x; dropout = P) |> Array
@test isapprox(mean(y.==0), P; atol = 3/sqrt(N))
@test y == cudnnDropoutForward(x, d) |> Array
@test y == cudnnDropoutForward!(similar(x), x; dropout = P) |> Array
@test y == cudnnDropoutForward!(similar(x), x, d) |> Array
cudnnDropoutSeed[] = -1

@testset "multiple tasks" begin
    # each task uses its own random number generator state, also when it is passed a
    # descriptor that was created by another task
    cudnnDropoutSeed[] = 1
    y = cudnnDropoutForward(x; dropout = P) |> Array
    @test y == fetch(Threads.@spawn Array(cudnnDropoutForward(x; dropout = P)))
    @test y == fetch(Threads.@spawn Array(cudnnDropoutForward(x, d)))
    cudnnDropoutSeed[] = -1

    # (the outer task keeps its handle, so that the inner task uses another one)
    @test fetch(Threads.@spawn begin
        h = cuDNN.handle()
        d = cudnnDropoutDescriptor(P)
        fetch(Threads.@spawn cuDNN.dropout_descriptor(cuDNN.handle(), d).state) !== d.state
    end)
end

@testset "graph capture" begin
    y = similar(x)
    cudnnDropoutForward!(y, x; dropout = P)
    graph = capture(() -> cudnnDropoutForward!(y, x; dropout = P))
    # the graph keeps the random number generator state alive
    @test cuDNN.handle().dropout.state.data[] in graph.memory
    exec = instantiate(graph)
    launch(exec)
    y1 = Array(y)
    @test isapprox(mean(y1.==0), P; atol = 3/sqrt(N))
    # replaying the graph advances the random number generator state
    launch(exec)
    @test Array(y) != y1
end
