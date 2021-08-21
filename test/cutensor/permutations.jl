using CUDA.CUTENSOR
using CUDA
using LinearAlgebra

# using host memory with CUTENSOR doesn't work on Windows
can_pin = !Sys.iswindows()

eltypes = ((Float16, Float16),
            #(Float16, Float32),
            (Float32, Float32),
            #(Float32, Float64),
            (Float64, Float64),
            #(ComplexF16, ComplexF16),
            #(ComplexF16, ComplexF32),
            (ComplexF32, ComplexF32),
            #(ComplexF32, ComplexF64),
            (ComplexF64, ComplexF64))
@testset for N=2:5
    @testset for (eltyA, eltyC) in eltypes
        # setup
        dmax = 2^div(18,N)
        dims = rand(2:dmax, N)
        p = randperm(N)
        indsA = collect(('a':'z')[1:N])
        indsC = indsA[p]
        dimsA = dims
        dimsC = dims[p]
        A = rand(eltyA, dimsA...)
        can_pin && Mem.pin(A)
        dA = CuArray(A)
        dC = similar(dA, eltyC, dimsC...)

        # simple case
        dC = CUTENSOR.permutation!(one(eltyA), dA, indsA, dC, indsC)
        C  = collect(dC)
        @test C == permutedims(A, p) # exact equality
        if can_pin
            Csimple = zeros(eltyC, dimsC...)
            Mem.pin(Csimple)
            Csimple = CUDA.@sync CUTENSOR.permutation!(one(eltyA), A, indsA, Csimple, indsC)
            @test Csimple == permutedims(A, p) # exact equality
        end

        # with scalar
        α  = rand(eltyA)
        dC = CUTENSOR.permutation!(α, dA, indsA, dC, indsC)
        C  = collect(dC)
        @test C ≈ α * permutedims(A, p) # approximate, floating point rounding
        if can_pin
            Cscalar = zeros(eltyC, dimsC...)
            Mem.pin(Cscalar)
            Cscalar = CUDA.@sync CUTENSOR.permutation!(α, A, indsA, Cscalar, indsC)
            @test Cscalar ≈ α * permutedims(A, p) # approximate, floating point rounding
        end
    end
end
