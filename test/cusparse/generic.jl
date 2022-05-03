using CUDA
using Adapt
using CUDA.CUSPARSE
using SparseArrays

if CUDA.version() ≥ v"11.3"
    @testset "generic mv!" for T in [Float32, Float64]
        A = sprand(T, 10, 10, 0.1)
        x = rand(Complex{T}, 10)
        y = similar(x)
        dx = adapt(CuArray, x)
        dy = adapt(CuArray, y)

        dA = adapt(CuArray, A)
        mv!('N', 1.0, dA, dx, 0.0, dy, 'O')
        @test Array(dy) ≈ A * x

        dA = CuSparseMatrixCSR(dA)
        mv!('N', 1.0, dA, dx, 0.0, dy, 'O')
        @test Array(dy) ≈ A * x
    end
end
