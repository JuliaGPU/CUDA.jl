@testset "contractions" begin

using cuTENSOR: contract!, plan_contraction

using LinearAlgebra

eltypes = [(Float32, Float32, Float32, Float32),
           (Float32, Float32, Float32, Float16),
           (Float16, Float16, Float16, Float32),
           (ComplexF32, ComplexF32, ComplexF32, Float32),
           (Float64, Float64, Float64, Float64),
           (Float64, Float64, Float64, Float32),
           (ComplexF64, ComplexF64, ComplexF64, Float64),
           (ComplexF64, ComplexF64, ComplexF64, Float32)]

@testset for NoA=1:2, NoB=1:2, Nc=1:2
    @testset for (eltyA, eltyB, eltyC, eltyCompute) in eltypes
        # setup
        dmax = 2^div(12, max(NoA+Nc, NoB+Nc, NoA+NoB))
        dimsoA = rand(2:dmax, NoA)
        loA = prod(dimsoA)
        dimsoB = rand(2:dmax, NoB)
        loB = prod(dimsoB)
        dimsc = rand(2:dmax, Nc)
        lc = prod(dimsc)
        allinds = collect('a':'z')
        indsoA = allinds[1:NoA]
        indsoB = allinds[NoA .+ (1:NoB)]
        indsc = allinds[NoA .+ NoB .+ (1:Nc)]
        pA = randperm(NoA + Nc)
        ipA = invperm(pA)
        pB = randperm(Nc + NoB)
        ipB = invperm(pB)
        pC = randperm(NoA + NoB)
        ipC = invperm(pC)
        compute_rtol = (eltyCompute == Float16 || eltyC == Float16) ? 1e-2 : (eltyCompute == Float32 ? 1e-4 : 1e-6)
        dimsA = [dimsoA; dimsc][pA]
        indsA = [indsoA; indsc][pA]
        dimsB = [dimsc; dimsoB][pB]
        indsB = [indsc; indsoB][pB]
        dimsC = [dimsoA; dimsoB][pC]
        indsC = [indsoA; indsoB][pC]

        A = rand(eltyA, (dimsA...,))
        mA = reshape(permutedims(A, ipA), (loA, lc))
        B = rand(eltyB, (dimsB...,))
        mB = reshape(permutedims(B, ipB), (lc, loB))
        C = zeros(eltyC, (dimsC...,))
        dA = CuArray(A)
        dB = CuArray(B)
        dC = CuArray(C)
        # simple case
        opA = cuTENSOR.OP_IDENTITY
        opB = cuTENSOR.OP_IDENTITY
        opC = cuTENSOR.OP_IDENTITY
        opOut = cuTENSOR.OP_IDENTITY
        dC = contract!(1, dA, indsA, opA, dB, indsB, opB, 0, dC, indsC, opC, opOut, compute_type=eltyCompute)
        C = collect(dC)
        mC = reshape(permutedims(C, ipC), (loA, loB))
        @test mC ≈ mA * mB rtol=compute_rtol

        # simple case with plan storage
        opA = cuTENSOR.OP_IDENTITY
        opB = cuTENSOR.OP_IDENTITY
        opC = cuTENSOR.OP_IDENTITY
        opOut = cuTENSOR.OP_IDENTITY
        plan  = cuTENSOR.plan_contraction(dA, indsA, opA, dB, indsB, opB, dC, indsC, opC, opOut)
        dC = cuTENSOR.contract!(plan, 1, dA, dB, 0, dC)
        C = collect(dC)
        mC = reshape(permutedims(C, ipC), (loA, loB))
        @test mC ≈ mA * mB

        # simple case with plan storage and compute type
        opA = cuTENSOR.OP_IDENTITY
        opB = cuTENSOR.OP_IDENTITY
        opC = cuTENSOR.OP_IDENTITY
        opOut = cuTENSOR.OP_IDENTITY
        eltypComputeEnum = convert(cuTENSOR.cutensorComputeDescriptorEnum, eltyCompute)
        plan  = cuTENSOR.plan_contraction(dA, indsA, opA, dB, indsB, opB, dC, indsC, opC, opOut; compute_type=eltypComputeEnum)
        dC = cuTENSOR.contract!(plan, 1, dA, dB, 0, dC)
        C = collect(dC)
        mC = reshape(permutedims(C, ipC), (loA, loB))
        @test mC ≈ mA * mB rtol=compute_rtol

        # simple case with plan storage and JIT compilation
        opA = cuTENSOR.OP_IDENTITY
        opB = cuTENSOR.OP_IDENTITY
        opC = cuTENSOR.OP_IDENTITY
        opOut = cuTENSOR.OP_IDENTITY
        plan  = cuTENSOR.plan_contraction(dA, indsA, opA, dB, indsB, opB, dC, indsC, opC, opOut; jit=cuTENSOR.JIT_MODE_DEFAULT)
        dC = cuTENSOR.contract!(plan, 1, dA, dB, 0, dC)
        C = collect(dC)
        mC = reshape(permutedims(C, ipC), (loA, loB))
        @test mC ≈ mA * mB

        # with non-trivial α
        α = rand(eltyCompute)
        dC = contract!(α, dA, indsA, opA, dB, indsB, opB, zero(eltyCompute), dC, indsC, opC, opOut; compute_type=eltyCompute)
        C = collect(dC)
        mC = reshape(permutedims(C, ipC), (loA, loB))
        @test mC ≈ α * mA * mB rtol=compute_rtol

        # with non-trivial β
        C = rand(eltyC, (dimsC...,))
        dC = CuArray(C)
        α = rand(eltyCompute)
        β = rand(eltyCompute)
        copyto!(dC, C)
        dD = contract!(α, dA, indsA, opA, dB, indsB, opB, β, dC, indsC, opC, opOut; compute_type=eltyCompute)
        D = collect(dD)
        mC = reshape(permutedims(C, ipC), (loA, loB))
        mD = reshape(permutedims(D, ipC), (loA, loB))
        @test mD ≈ α * mA * mB + β * mC rtol=compute_rtol
        # with CuTensor objects
        if eltyCompute != Float32 && eltyC != Float16
            ctA = CuTensor(dA, indsA)
            ctB = CuTensor(dB, indsB)
            ctC = CuTensor(dC, indsC)
            ctC = LinearAlgebra.mul!(ctC, ctA, ctB)
            C2, C2inds = collect(ctC)
            mC = reshape(permutedims(C2, ipC), (loA, loB))
            @test mC ≈ mA * mB
            ctC = ctA * ctB
            C2, C2inds = collect(ctC)
            pC2 = Int.(indexin(Char.(C2inds), [indsoA; indsoB]))
            mC = reshape(permutedims(C2, invperm(pC2)), (loA, loB))
            @test mC ≈ mA * mB
        end

        # with conjugation flag for complex arguments
        if !((NoA, NoB, Nc) in ((1,1,3), (1,2,3), (3,1,2)))
        # not supported for these specific cases for unknown reason
            if eltyA <: Complex
                opA   = cuTENSOR.OP_CONJ
                opB   = cuTENSOR.OP_IDENTITY
                opOut = cuTENSOR.OP_IDENTITY
                dC    = contract!(complex(1.0, 0.0), dA, indsA, opA, dB, indsB, opB,
                                                0, dC, indsC, opC, opOut; compute_type=eltyCompute)
                C     = collect(dC)
                mC    = reshape(permutedims(C, ipC), (loA, loB))
                @test mC ≈ conj(mA) * mB rtol=compute_rtol
            end
            if eltyB <: Complex
                opA = cuTENSOR.OP_IDENTITY
                opB = cuTENSOR.OP_CONJ
                opOut = cuTENSOR.OP_IDENTITY
                dC = contract!(complex(1.0, 0.0), dA, indsA, opA, dB, indsB, opB,
                               complex(0.0, 0.0), dC, indsC, opC, opOut; compute_type=eltyCompute)
                C = collect(dC)
                mC = reshape(permutedims(C, ipC), (loA, loB))
                @test mC ≈ mA*conj(mB) rtol=compute_rtol
            end
            if eltyA <: Complex && eltyB <: Complex
                opA = cuTENSOR.OP_CONJ
                opB = cuTENSOR.OP_CONJ
                opOut = cuTENSOR.OP_IDENTITY
                dC = contract!(one(eltyCompute), dA, indsA, opA, dB, indsB, opB,
                        zero(eltyCompute), dC, indsC, opC, opOut; compute_type=eltyCompute)
                C = collect(dC)
                mC = reshape(permutedims(C, ipC), (loA, loB))
                @test mC ≈ conj(mA)*conj(mB) rtol=compute_rtol
            end
        end
    end
end

# https://github.com/JuliaGPU/CUDA.jl/issues/2407
@testset "contractions of views" begin
    @testset for (eltyA, eltyB, eltyC, eltyCompute) in eltypes
        dimsA = (16,)
        dimsB = (4,)
        dimsC = (8,)
        A = rand(eltyA, dimsA)
        B = rand(eltyB, dimsB)
        C = rand(eltyC, dimsC)
        dA = CuArray(A)
        dB = CuArray(B)
        dC = CuArray(C)
        dD = CuArray(C)
        vA = @view dA[1:4]
        vB = @view dB[4:4]
        vC = @view dC[3:6]
        vD = @view dD[3:6]
        tA = CuTensor(reshape(vA, (4, 1)), [1, 2])
        tB = CuTensor(reshape(vB, (1, 1)), [3, 2])
        tC = CuTensor(reshape(vC, (1, 4)), [3, 1])
        mul!(tC, tA, tB)
    end
end

end
