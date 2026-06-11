using cuSOLVER
using LinearAlgebra

n = 10

@testset "Hermitian/Symmetric matrix functions, elty = $elty" for elty in [Float32, Float64, ComplexF32, ComplexF64]
    A = rand(elty, n, n)
    Ah = A * A' # make posdef for atan, asinh, atanh
    d_Ah = CuArray(Ah)
    @testset for func in (exp, cos, sin, tan, cosh, sinh, tanh, atan, asinh)
        @test Array(func(d_Ah)) ≈ func(Ah)
    end
    @test Array(parent(log(Hermitian(d_Ah)))) ≈ log(Hermitian(Ah))
    if elty <: Real
        @test Array(parent(log(Symmetric(d_Ah)))) ≈ log(Symmetric(Ah))
    end
    @static if VERSION >= v"1.11.0" # not supported on 1.10 or for Complex
        if elty <: Real
            @testset for func in (cbrt,) # have to dispatch explicitly
                @test Array(parent(func(Hermitian(d_Ah)))) ≈ func(Ah)
            end
        end
    end
end
