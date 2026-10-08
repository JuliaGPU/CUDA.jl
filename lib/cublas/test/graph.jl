using cuBLAS
using LinearAlgebra

@testset "graph capture" begin
    # scalar arguments are passed by reference, using memory that's copied from the CPU
    # (JuliaGPU/CUDA.jl#2691)
    N, M = 20, 10
    A = CuArray(reshape(LinRange(0, 1, N*N*M), N, N, M))
    x = CuArray(reshape(LinRange(0, 1, N*M), N, M))
    y = similar(x)
    cuBLAS.gemv_strided_batched!('N', 1, A, x, 0, y)
    expected = Array(y)

    y .= 0
    exec = instantiate(capture(() -> cuBLAS.gemv_strided_batched!('N', 1, A, x, 0, y)))
    for _ in 1:3
        y .= 0
        exec()
        @test Array(y) ≈ expected
    end

    # higher-level operations, allocating their output
    a = CUDA.rand(Float32, 16, 16)
    b = CUDA.rand(Float32, 16, 16)
    a * b
    local c
    exec = instantiate(capture(() -> c = a * b))
    for _ in 1:3
        copyto!(a, rand(Float32, 16, 16))
        exec()
        @test Array(c) ≈ Array(a) * Array(b)
    end
end

@testset "tasks spawned during capture" begin
    A = CUDA.rand(Float32, 16, 16)
    B = CUDA.rand(Float32, 16, 16)
    C1 = similar(A)
    C2 = similar(A)
    mul!(C1, A, B)

    # tasks that already used cuBLAS (with handles bound to their own streams)
    worker = Threads.@spawn begin
        mul!(C2, B, A)
        synchronize()
    end
    wait(worker)

    exec = instantiate(capture() do
        @sync begin
            Threads.@spawn mul!(C1, A, B)
            Threads.@spawn mul!(C2, B, A)
        end
        C1 .+= C2
    end)
    for _ in 1:3
        copyto!(A, rand(Float32, 16, 16))
        copyto!(B, rand(Float32, 16, 16))
        exec()
        @test Array(C1) ≈ Array(A) * Array(B) + Array(B) * Array(A)
    end
end
