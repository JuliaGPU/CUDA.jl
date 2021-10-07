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

@testset "ldiv!" begin
    A = rand(Float32, 2, 2)
    Q = qr(A)
    _Q = qr(cu(A))
    x = rand(Float32, 2)
    _x = cu(x)
    y = similar(x)
    _y = similar(_x)

    ldiv!(y,Q,x)
    ldiv!(_y,_Q,_x)
    @test y ≈ Array(_y)

    ldiv!(Q,x)
    ldiv!(_Q,_x)
    @test x ≈ Array(_x)
end
