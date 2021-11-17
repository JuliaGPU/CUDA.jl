using LinearAlgebra

@testset "qr size mismatch" begin
    X = rand(Float32, 2, 1)
    Q,R = qr(X)

    @test collect(Q) == Array(collect(Q))
    @test Array(Q) == Array(CuArray(Q))
    @test Array{Float32}(Q) == Array(CuArray{Float32}(Q))
    @test Matrix(Q) == Array(CuMatrix(Q))
    @test Matrix{Float32}(Q) == Array(CuMatrix{Float32}(Q))
    @test convert(Array, Q) == Array(convert(CuArray, Q))
    @test convert(Array{Float32}, Q) == Array(convert(CuArray{Float32}, Q))
end

@testset "normalize!" begin
    x = rand(ComplexF32, 10)
    dx = CuVector{ComplexF32}(x)
    @test isreal(norm(dx, 2))
    @test norm(normalize!(dx)) ≈ 1
end

@testset "dot" begin
    @testset for T in [Int16, Int32, Int64,
                       Float16, Float32, Float64,
                       #=ComplexF16, ComplexF32, ComplexF64=#]
        # TODO: complex types aren't supported by @atomic
        # - ComplexF16/ComplexF32 can't be bitcasted to integers (JuliaLang/julia#43065)
        # - 128-bit datatypes are not supported by a single atomic (can we split the operation?)
        @test testf(dot, rand(T, 256), rand(Bool, 256))
        @test testf(dot, rand(Bool, 256), rand(T, 256))
    end

    @test testf(dot, rand(Bool, 1024, 1024), rand(Float64, 1024, 1024))
end
