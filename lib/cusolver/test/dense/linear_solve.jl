using cuSOLVER
using LinearAlgebra

m = 15
n = 10
p = 5

@testset "gesv! elty = $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64]
    @testset "irs_precision = AUTO" begin
        A = rand(elty, n, n)
        X = zeros(elty, n, p)
        B = rand(elty, n, p)
        dA = CuArray(A)
        dX = CuArray(X)
        dB = CuArray(B)
        cuSOLVER.gesv!(dX, dA, dB)
        tol = real(elty) |> eps |> sqrt
        dR = dB - dA * dX
        @test norm(dR) <= tol
    end
    @testset "irs_precision = $elty" begin
        irs_precision = elty <: Real ? "R_" : "C_"
        irs_precision *= string(sizeof(real(elty)) * 8) * "F"
        A = rand(elty, n, n)
        X = zeros(elty, n, p)
        B = rand(elty, n, p)
        dA = CuArray(A)
        dX = CuArray(X)
        dB = CuArray(B)
        cuSOLVER.gesv!(dX, dA, dB; irs_precision=irs_precision)
        tol = real(elty) |> eps |> sqrt
        dR = dB - dA * dX
        @test norm(dR) <= tol
    end
    @testset "IRSParameters" begin
        params = cuSOLVER.CuSolverIRSParameters()
        max_iter = 10
        cuSOLVER.cusolverDnIRSParamsSetMaxIters(params, max_iter)
        @test cuSOLVER.get_info(params, :maxiters) == max_iter
        @test_throws ErrorException("The information fake is incorrect.") cuSOLVER.get_info(params, :fake)
        A = rand(elty, n, n)
        X = zeros(elty, n, p)
        B = rand(elty, n, p)
        dA = CuArray(A)
        dX = CuArray(X)
        dB = CuArray(B)
        dX, info = cuSOLVER.gesv!(dX, dA, dB; maxiters=max_iter)
        @test cuSOLVER.get_info(info, :maxiters) == max_iter
        @test cuSOLVER.get_info(info, :niters) <= max_iter
        @test cuSOLVER.get_info(info, :outer_niters) <= max_iter
        @test_throws ErrorException("The information fake is incorrect.") cuSOLVER.get_info(info, :fake)
    end
end

@testset "gels! elty = $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64]
    @testset "irs_precision = AUTO" begin
        A = rand(elty, m, n)
        X = zeros(elty, n, p)
        B = A * rand(elty, n, p)  # ensure that AX = B is consistent
        dA = CuArray(A)
        dX = CuArray(X)
        dB = CuArray(B)
        cuSOLVER.gels!(dX, dA, dB)
        tol = real(elty) |> eps |> sqrt
        dR = dB - dA * dX
        @test norm(dR) <= tol
    end
    @testset "irs_precision = $elty" begin
        irs_precision = elty <: Real ? "R_" : "C_"
        irs_precision *= string(sizeof(real(elty)) * 8) * "F"
        A = rand(elty, m, n)
        X = zeros(elty, n, p)
        B = A * rand(elty, n, p)  # ensure that AX = B is consistent
        dA = CuArray(A)
        dX = CuArray(X)
        dB = CuArray(B)
        cuSOLVER.gels!(dX, dA, dB; irs_precision=irs_precision)
        tol = real(elty) |> eps |> sqrt
        dR = dB - dA * dX
        @test norm(dR) <= tol
    end
end
