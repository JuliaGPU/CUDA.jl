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
