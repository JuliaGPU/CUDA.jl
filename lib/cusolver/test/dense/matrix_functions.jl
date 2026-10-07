using cuSOLVER
using LinearAlgebra

n = 10

# all matrix functions are generated from a single `@eval` loop over the Symmetric and
# Hermitian wrappers, so one real and one complex element type cover every method.
# (the Hermitian eigenvalues of ComplexF32 matrices are Float32, so between them the
# two element types also apply each scalar function to Float32 and Float64 values.)
@testset "Hermitian/Symmetric matrix functions, elty = $elty" for elty in [Float64, ComplexF32]
    A = rand(elty, n, n)
    Ah = A * A' # make posdef for atan, asinh, atanh
    d_Ah = CuArray(Ah)
    @testset for func in (exp, cos, sin, tan, cosh, sinh, tanh, atan, asinh)
        @test Array(parent(func(Hermitian(d_Ah)))) ≈ func(Hermitian(Ah))
    end
    # plain matrices dispatch to the Hermitian/Symmetric methods through LinearAlgebra
    @testset for func in (exp, cos)
        @test Array(func(d_Ah)) ≈ func(Ah)
        if elty <: Real
            @test Array(parent(func(Symmetric(d_Ah)))) ≈ func(Symmetric(Ah))
        end
    end
    @test Array(parent(log(Hermitian(d_Ah)))) ≈ log(Hermitian(Ah))
    if elty <: Real
        @test Array(parent(log(Symmetric(d_Ah)))) ≈ log(Symmetric(Ah))
    end
end

@static if VERSION >= v"1.11.0" # not supported on 1.10 or for Complex
    @testset "cbrt, elty = $elty" for elty in [Float32, Float64] # have to dispatch explicitly
        A = rand(elty, n, n)
        Ah = A * A'
        d_Ah = CuArray(Ah)
        @test Array(parent(cbrt(Hermitian(d_Ah)))) ≈ cbrt(Ah)
    end
end
