@run_if has_cutensor() && CUDA.version() >= v"10.1" && capability(device()) >= v"7.0" begin
@not_if_memcheck begin

using CUDA.CUTENSOR
using CUDA
using LinearAlgebra
using Random

@testcase "CUTENSOR" begin

# using host memory with CUTENSOR doesn't work on Windows
can_pin = !Sys.iswindows()

# TODO: split into more find-grained test cases

############################################################################################

@testcase "type basics" begin
    @test CUTENSOR.version() isa VersionNumber

    N = 2
    dmax = 2^div(18,N)
    dims = rand(2:dmax, N)
    p = randperm(N)
    indsA = collect(('a':'z')[1:N])
    dimsA = dims
    A = rand(Float64, dimsA...)
    dA = CuArray(A)
    p = randperm(N)
    indsA = collect(('a':'z')[1:N])
    ctA = CuTensor(dA, indsA)
    @test length(ctA) == length(A)
    @test size(ctA) == size(A)
    @test size(ctA, 1) == size(A, 1)
    @test ndims(ctA) == ndims(A)
    @test strides(ctA) == strides(A)
    @test eltype(ctA) == eltype(A)
end

############################################################################################

@testset "contractions" begin

eltypes = ( (Float32, Float32, Float32, Float32),
            (Float32, Float32, Float32, Float16),
            (ComplexF32, ComplexF32, ComplexF32, ComplexF32),
            (Float64, Float64, Float64, Float64),
            (Float64, Float64, Float64, Float32),
            (ComplexF64, ComplexF64, ComplexF64, ComplexF64),
            (ComplexF64, ComplexF64, ComplexF64, ComplexF32)
            )

@testset for NoA=1:3, NoB=1:3, Nc=1:3
    @testcase for (eltyA, eltyB, eltyC, eltyCompute) in eltypes
        # setup
        dmax = 2^div(18, max(NoA+Nc, NoB+Nc, NoA+NoB))
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
        compute_rtol = (real(eltyCompute) == Float16 || real(eltyC) == Float16) ? 1e-2 : (real(eltyCompute) == Float32 ? 1e-4 : 1e-6)
        dimsA = [dimsoA; dimsc][pA]
        indsA = [indsoA; indsc][pA]
        dimsB = [dimsc; dimsoB][pB]
        indsB = [indsc; indsoB][pB]
        dimsC = [dimsoA; dimsoB][pC]
        indsC = [indsoA; indsoB][pC]

        A = rand(eltyA, (dimsA...,))
        can_pin && Mem.pin(A)
        dA = CuArray(A)
        mA = reshape(permutedims(A, ipA), (loA, lc))
        B = rand(eltyB, (dimsB...,))
        can_pin && Mem.pin(B)
        dB = CuArray(B)
        mB = reshape(permutedims(B, ipB), (lc, loB))
        C = zeros(eltyC, (dimsC...,))
        dC = CuArray(C)

        # simple case host side
        if can_pin
            Cpin = zeros(eltyC, (dimsC...,))
            Mem.pin(Cpin)
            @test !any(isnan.(Cpin))
            opA = CUTENSOR.CUTENSOR_OP_IDENTITY
            opB = CUTENSOR.CUTENSOR_OP_IDENTITY
            opC = CUTENSOR.CUTENSOR_OP_IDENTITY
            opOut = CUTENSOR.CUTENSOR_OP_IDENTITY
            Cpin  = CUDA.@sync CUTENSOR.contraction!(1, A, indsA, opA, B, indsB, opB, 0, Cpin, indsC, opC, opOut)
            mCpin = reshape(permutedims(Cpin, ipC), (loA, loB))
            @test !any(isnan.(A))
            @test !any(isnan.(B))
            @test !any(isnan.(mCpin))
            @test mCpin ≈ mA * mB rtol=compute_rtol
        end

        # simple case
        opA = CUTENSOR.CUTENSOR_OP_IDENTITY
        opB = CUTENSOR.CUTENSOR_OP_IDENTITY
        opC = CUTENSOR.CUTENSOR_OP_IDENTITY
        opOut = CUTENSOR.CUTENSOR_OP_IDENTITY
        dC = CUTENSOR.contraction!(1, dA, indsA, opA, dB, indsB, opB, 0, dC, indsC, opC, opOut, compute_type=eltyCompute)
        C = collect(dC)
        mC = reshape(permutedims(C, ipC), (loA, loB))
        @test mC ≈ mA * mB rtol=compute_rtol

        # simple case with plan storage
        opA = CUTENSOR.CUTENSOR_OP_IDENTITY
        opB = CUTENSOR.CUTENSOR_OP_IDENTITY
        opC = CUTENSOR.CUTENSOR_OP_IDENTITY
        opOut = CUTENSOR.CUTENSOR_OP_IDENTITY
        plan  = CUTENSOR.plan_contraction(dA, indsA, opA, dB, indsB, opB, dC, indsC, opC, opOut)
        dC = CUTENSOR.contraction!(1, dA, indsA, opA, dB, indsB, opB, 0, dC, indsC, opC, opOut, plan=plan)
        C = collect(dC)
        mC = reshape(permutedims(C, ipC), (loA, loB))
        @test mC ≈ mA * mB

        # simple case with plan storage host-side
        if can_pin
            opA = CUTENSOR.CUTENSOR_OP_IDENTITY
            opB = CUTENSOR.CUTENSOR_OP_IDENTITY
            opC = CUTENSOR.CUTENSOR_OP_IDENTITY
            opOut = CUTENSOR.CUTENSOR_OP_IDENTITY
            plan  = CUTENSOR.plan_contraction(A, indsA, opA, B, indsB, opB, C, indsC, opC, opOut)
            Cpin = CUDA.@sync CUTENSOR.contraction!(1, A, indsA, opA, B, indsB, opB, 0, Cpin, indsC, opC, opOut, plan=plan)
            mC = reshape(permutedims(Cpin, ipC), (loA, loB))
            @test !any(isnan.(mC))
            @test mC ≈ mA * mB
        end

        # simple case with plan storage and compute type
        opA = CUTENSOR.CUTENSOR_OP_IDENTITY
        opB = CUTENSOR.CUTENSOR_OP_IDENTITY
        opC = CUTENSOR.CUTENSOR_OP_IDENTITY
        opOut = CUTENSOR.CUTENSOR_OP_IDENTITY
        plan  = CUTENSOR.plan_contraction(dA, indsA, opA, dB, indsB, opB, dC, indsC, opC, opOut, compute_type=eltyCompute)
        dC = CUTENSOR.contraction!(1, dA, indsA, opA, dB, indsB, opB,
                                    0, dC, indsC, opC, opOut, plan=plan, compute_type=eltyCompute)
        C = collect(dC)
        mC = reshape(permutedims(C, ipC), (loA, loB))
        @test mC ≈ mA * mB rtol=compute_rtol

        # with non-trivial α
        α = rand(eltyCompute)
        dC = CUTENSOR.contraction!(α, dA, indsA, opA, dB, indsB, opB, zero(eltyCompute), dC, indsC, opC, opOut, compute_type=eltyCompute)
        C = collect(dC)
        mC = reshape(permutedims(C, ipC), (loA, loB))
        @test mC ≈ α * mA * mB rtol=compute_rtol

        if can_pin
            α = rand(eltyCompute)
            Calpha = zeros(eltyC, (dimsC...,))
            Mem.pin(Calpha)
            @test !any(isnan.(Calpha))
            Calpha = CUDA.@sync CUTENSOR.contraction!(α, A, indsA, opA, B, indsB, opB, 0, Calpha, indsC, opC, opOut)
            mCalpha = reshape(permutedims(collect(Calpha), ipC), (loA, loB))
            @test !any(isnan.(mCalpha))
            @test mCalpha ≈ α * mA * mB rtol=compute_rtol
        end

        # with non-trivial β
        C = rand(eltyC, (dimsC...,))
        dC = CuArray(C)
        α = rand(eltyCompute)
        β = rand(eltyCompute)
        copyto!(dC, C)
        dD = CUTENSOR.contraction!(α, dA, indsA, opA, dB, indsB, opB, β, dC, indsC, opC, opOut, compute_type=eltyCompute)
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
            pC2 = convert.(Int, indexin(convert.(Char, C2inds), [indsoA; indsoB]))
            mC = reshape(permutedims(C2, invperm(pC2)), (loA, loB))
            @test mC ≈ mA * mB
        end
        # with conjugation flag for complex arguments
        if !((NoA, NoB, Nc) in ((1,1,3), (1,2,3), (3,1,2)))
        # not supported for these specific cases for unknown reason
            if eltyA <: Complex
                opA   = CUTENSOR.CUTENSOR_OP_CONJ
                opB   = CUTENSOR.CUTENSOR_OP_IDENTITY
                opOut = CUTENSOR.CUTENSOR_OP_IDENTITY
                dC    = CUTENSOR.contraction!(complex(1.0, 0.0), dA, indsA, opA, dB, indsB, opB,
                                                0, dC, indsC, opC, opOut, compute_type=eltyCompute)
                C     = collect(dC)
                mC    = reshape(permutedims(C, ipC), (loA, loB))
                @test mC ≈ conj(mA) * mB rtol=compute_rtol
            end
            if eltyB <: Complex
                opA = CUTENSOR.CUTENSOR_OP_IDENTITY
                opB = CUTENSOR.CUTENSOR_OP_CONJ
                opOut = CUTENSOR.CUTENSOR_OP_IDENTITY
                dC = CUTENSOR.contraction!(complex(1.0, 0.0), dA, indsA, opA, dB, indsB, opB,
                                            complex(0.0, 0.0), dC, indsC, opC, opOut, compute_type=eltyCompute)
                C = collect(dC)
                mC = reshape(permutedims(C, ipC), (loA, loB))
                @test mC ≈ mA*conj(mB) rtol=compute_rtol
            end
            if eltyA <: Complex && eltyB <: Complex
                opA = CUTENSOR.CUTENSOR_OP_CONJ
                opB = CUTENSOR.CUTENSOR_OP_CONJ
                opOut = CUTENSOR.CUTENSOR_OP_IDENTITY
                dC = CUTENSOR.contraction!(one(eltyCompute), dA, indsA, opA, dB, indsB, opB,
                        zero(eltyCompute), dC, indsC, opC, opOut, compute_type=eltyCompute)
                C = collect(dC)
                mC = reshape(permutedims(C, ipC), (loA, loB))
                @test mC ≈ conj(mA)*conj(mB) rtol=compute_rtol
            end
        end
    end
end

end

############################################################################################

@testset "elementwise binary" begin

eltypes = ((Float16, Float16),
            #(Float16, Float32),
            (Float32, Float32),
            #(Float32, Float64),
            (Float64, Float64),
            #(ComplexF16, ComplexF16), (ComplexF16, ComplexF32),
            (ComplexF32, ComplexF32), #(ComplexF32, ComplexF64),
            (ComplexF64, ComplexF64))

@testset for N=2:5
    @testcase for (eltyA, eltyC) in eltypes
        # setup
        eltyD = eltyC
        dmax  = 2^div(18,N)
        dims  = rand(2:dmax, N)
        p     = randperm(N)
        indsA = collect(('a':'z')[1:N])
        indsC = indsA[p]
        dimsA = dims
        dimsC = dims[p]
        A     = rand(eltyA, dimsA...)
        dA    = CuArray(A)
        C     = rand(eltyC, dimsC...)
        dC    = CuArray(C)
        can_pin && Mem.pin(A)
        can_pin && Mem.pin(C)

        # simple case
        opA   = CUTENSOR.CUTENSOR_OP_IDENTITY
        opC   = CUTENSOR.CUTENSOR_OP_IDENTITY
        dD    = similar(dC, eltyD)
        opAC  = CUTENSOR.CUTENSOR_OP_ADD
        dD    = CUTENSOR.elementwiseBinary!(1, dA, indsA, opA, 1, dC, indsC, opC,
                                            dD, indsC, opAC)
        D = collect(dD)
        @test D ≈ permutedims(A, p) .+ C
        if can_pin
            Dsimple = similar(C)
             Mem.pin(Dsimple)
            Dsimple = CUDA.@sync CUTENSOR.elementwiseBinary!(1, A, indsA, opA, 1, C, indsC, opC,
                                                             Dsimple, indsC, opAC)
            @test Dsimple ≈ permutedims(A, p) .+ C
        end

        # using integers as indices
        dD = CUTENSOR.elementwiseBinary!(1, dA, 1:N, opA, 1, dC, p, opC, dD, p, opAC)
        D = collect(dD)
        @test D ≈ permutedims(A, p) .+ C
        if can_pin
            Dint = zeros(eltyC, dimsC...)
            Mem.pin(Dint)
            Dint = CUDA.@sync CUTENSOR.elementwiseBinary!(1, A, 1:N, opA, 1, C, p, opC, Dint, p, opAC)
            @test Dint ≈ permutedims(A, p) .+ C
        end

        # multiplication as binary operator
        opAC = CUTENSOR.CUTENSOR_OP_MUL
        dD = CUTENSOR.elementwiseBinary!(1, dA, indsA, opA, 1, dC, indsC, opC,
                                            dD, indsC, opAC)
        D = collect(dD)
        @test D ≈ permutedims(A, p) .* C
        if can_pin
            Dmult = zeros(eltyC, dimsC...)
            Mem.pin(Dmult)
            Dmult = CUDA.@sync CUTENSOR.elementwiseBinary!(1, A, indsA, opA, 1, C, indsC, opC,
                                                           Dmult, indsC, opAC)
            @test Dmult ≈ permutedims(A, p) .* C
        end

        # with non-trivial coefficients and conjugation
        opA = eltyA <: Complex ? CUTENSOR.CUTENSOR_OP_CONJ :
                                CUTENSOR.CUTENSOR_OP_IDENTITY
        opC = CUTENSOR.CUTENSOR_OP_IDENTITY
        opAC = CUTENSOR.CUTENSOR_OP_ADD
        α = rand(eltyD)
        γ = rand(eltyD)
        dD = CUTENSOR.elementwiseBinary!(α, dA, indsA, opA, γ, dC, indsC, opC,
                                            dD, indsC, opAC)
        D = collect(dD)
        @test D ≈ α .* conj.(permutedims(A, p)) .+ γ .* C
        if can_pin
            Dnontrivial = similar(C)
            Mem.pin(Dnontrivial)
            Dnontrivial = CUDA.@sync CUTENSOR.elementwiseBinary!(α, A, indsA, opA, γ, C, indsC, opC,
                                                                 Dnontrivial, indsC, opAC)
            @test Dnontrivial ≈ α .* conj.(permutedims(A, p)) .+ γ .* C
        end

        # test in-place, and more complicated unary and binary operations
        opA = eltyA <: Complex ? CUTENSOR.CUTENSOR_OP_IDENTITY :
                                CUTENSOR.CUTENSOR_OP_SQRT
        # because we use rand, entries of A will be positive when elty is real
        opC = eltyC <: Complex ? CUTENSOR.CUTENSOR_OP_CONJ :
                                CUTENSOR.CUTENSOR_OP_IDENTITY
        opAC = eltyD <: Complex ? CUTENSOR.CUTENSOR_OP_ADD :
                                CUTENSOR.CUTENSOR_OP_MAX
        α = rand(eltyD)
        γ = rand(eltyD)
        dD = CUTENSOR.elementwiseBinary!(α, dA, indsA, opA, γ, dC, indsC, opC,
                                            dC, indsC, opAC)
        D = collect(dC)
        if eltyD <: Complex
            if eltyA <: Complex
                @test D ≈ α .* permutedims(A, p) .+ γ .* conj.(C)
            else
                @test D ≈ α .* sqrt.(convert.(eltyD, permutedims(A, p))) .+
                            γ .* conj.(C)
            end
        else
            @test D ≈ max.(α .* sqrt.(convert.(eltyD, permutedims(A, p))), γ .* C)
        end

        # using CuTensor type
        dA = CuArray(A)
        dC = CuArray(C)
        ctA = CuTensor(dA, indsA)
        ctC = CuTensor(dC, indsC)
        ctD = ctA + ctC
        hD = collect(ctD.data)
        @test hD ≈ permutedims(A, p) .+ C
        ctD = ctA - ctC
        hD = collect(ctD.data)
        @test hD ≈ permutedims(A, p) .- C

        α = rand(eltyD)
        ctD = LinearAlgebra.axpy!(α, ctA, ctC)
        hD = collect(ctD.data)
        @test hD ≈ α.*permutedims(A, p) .+ C

        γ = rand(eltyD)
        ctD = LinearAlgebra.axpby!(α, ctA, γ, ctC)
        hD = collect(ctD.data)
        @test hD ≈ α.*permutedims(A, p) .+ γ.*C
    end
end

end

############################################################################################

@testset "elementwise trinary" begin

eltypes = ((Float16, Float16, Float16),
            #(Float16, Float32, Float32),
            # (Float32, Float16, Float32),
            (Float32, Float32, Float32),
            # (Float32, Float32, Float64),
            # (Float32, Float64, Float64),
            # (Float64, Float32, Float64),
            (Float64, Float64, Float64),
            (ComplexF32, ComplexF32, ComplexF32),
            (ComplexF64, ComplexF64, ComplexF64))

@testset for N=2:5
    @testcase for (eltyA, eltyB, eltyC) in eltypes
        # setup
        eltyD = eltyC
        dmax = 2^div(18,N)
        dims = rand(2:dmax, N)
        pA = randperm(N)
        ipA = invperm(pA)
        pB = randperm(N)
        ipB = invperm(pB)
        indsC = collect(('a':'z')[1:N])
        dimsC = dims
        indsA = indsC[ipA]
        dimsA = dims[ipA]
        indsB = indsC[ipB]
        dimsB = dims[ipB]
        A = rand(eltyA, dimsA...)
        dA = CuArray(A)
        B = rand(eltyB, dimsB...)
        dB = CuArray(B)
        C = rand(eltyC, dimsC...)
        dC = CuArray(C)
        dD = similar(dC)

        # simple case
        opA = CUTENSOR.CUTENSOR_OP_IDENTITY
        opB = CUTENSOR.CUTENSOR_OP_IDENTITY
        opC = CUTENSOR.CUTENSOR_OP_IDENTITY
        opAB = CUTENSOR.CUTENSOR_OP_ADD
        opABC = CUTENSOR.CUTENSOR_OP_ADD
        dD = CUTENSOR.elementwiseTrinary!(1, dA, indsA, opA, 1, dB, indsB, opB,
                                            1, dC, indsC, opC, dD, indsC, opAB, opABC)
        D = collect(dD)
        @test D ≈ permutedims(A, pA) .+ permutedims(B, pB) .+ C

        # test with Mem.pinned host memory
        if can_pin
            Mem.pin(A)
            Mem.pin(B)
            Mem.pin(C)
            Dc = similar(C)
            Mem.pin(Dc)
            Dc = CUDA.@sync CUTENSOR.elementwiseTrinary!(1, A, indsA, opA, 1, B, indsB, opB,
                                                         1, C, indsC, opC, Dc, indsC, opAB, opABC)
            @test Dc ≈ permutedims(A, pA) .+ permutedims(B, pB) .+ C
        end

        # using integers as indices
        dD = CUTENSOR.elementwiseTrinary!(1, dA, ipA, opA, 1, dB, ipB, opB,
                                            1, dC, 1:N, opC, dD, 1:N, opAB, opABC)
        D = collect(dD)
        @test D ≈ permutedims(A, pA) .+ permutedims(B, pB) .+ C
        if can_pin
            Dd = similar(C)
            Mem.pin(Dd)
            Dd = CUDA.@sync CUTENSOR.elementwiseTrinary!(1, A, ipA, opA, 1, B, ipB, opB,
                                                         1, C, 1:N, opC, Dd, 1:N, opAB, opABC)
            @test Dd ≈ permutedims(A, pA) .+ permutedims(B, pB) .+ C
        end

        # multiplication as binary operator
        opAB = CUTENSOR.CUTENSOR_OP_MUL
        opABC = CUTENSOR.CUTENSOR_OP_ADD
        dD = CUTENSOR.elementwiseTrinary!(1, dA, indsA, opA, 1, dB, indsB, opB,
                                            1, dC, indsC, opC, dD, indsC, opAB, opABC)
        D = collect(dD)
        @test D ≈ (convert.(eltyD, permutedims(A, pA)) .* convert.(eltyD, permutedims(B, pB))) .+ C
        if can_pin
            De = similar(C)
            Mem.pin(De)
            De = CUDA.@sync CUTENSOR.elementwiseTrinary!(1, A, indsA, opA, 1, B, indsB, opB,
                                                         1, C, indsC, opC, De, indsC, opAB, opABC)
            @test De ≈ (convert.(eltyD, permutedims(A, pA)) .* convert.(eltyD, permutedims(B, pB))) .+ C
        end

        opAB = CUTENSOR.CUTENSOR_OP_ADD
        opABC = CUTENSOR.CUTENSOR_OP_MUL
        dD = CUTENSOR.elementwiseTrinary!(1, dA, indsA, opA, 1, dB, indsB, opB,
                                            1, dC, indsC, opC, dD, indsC, opAB, opABC)
        D = collect(dD)
        @test D ≈ (convert.(eltyD, permutedims(A, pA)) .+ convert.(eltyD, permutedims(B, pB))) .* C
        if can_pin
            Df = similar(C)
            Mem.pin(Df)
            Df = CUDA.@sync CUTENSOR.elementwiseTrinary!(1, A, indsA, opA, 1, B, indsB, opB,
                                                         1, C, indsC, opC, Df, indsC, opAB, opABC)
            @test Df ≈ (convert.(eltyD, permutedims(A, pA)) .+ convert.(eltyD, permutedims(B, pB))) .* C
        end

        opAB = CUTENSOR.CUTENSOR_OP_MUL
        opABC = CUTENSOR.CUTENSOR_OP_MUL
        dD = CUTENSOR.elementwiseTrinary!(1, dA, indsA, opA, 1, dB, indsB, opB,
                                            1, dC, indsC, opC, dD, indsC, opAB, opABC)
        D = collect(dD)
        @test D ≈ convert.(eltyD, permutedims(A, pA)) .*
                    convert.(eltyD, permutedims(B, pB)) .* C
        if can_pin
            Dg = similar(C)
            Mem.pin(Dg)
            Dg = CUDA.@sync CUTENSOR.elementwiseTrinary!(1, A, indsA, opA, 1, B, indsB, opB,
                                                         1, C, indsC, opC, Dg, indsC, opAB, opABC)
            @test Dg ≈ convert.(eltyD, permutedims(A, pA)) .*
                       convert.(eltyD, permutedims(B, pB)) .* C
        end

        # with non-trivial coefficients and conjugation
        α = rand(eltyD)
        β = rand(eltyD)
        γ = rand(eltyD)
        opA = eltyA <: Complex ? CUTENSOR.CUTENSOR_OP_CONJ :
                                CUTENSOR.CUTENSOR_OP_IDENTITY
        opAB = CUTENSOR.CUTENSOR_OP_ADD
        opABC = CUTENSOR.CUTENSOR_OP_ADD
        dD = CUTENSOR.elementwiseTrinary!(α, dA, indsA, opA, β, dB, indsB, opB,
                                            γ, dC, indsC, opC, dD, indsC, opAB, opABC)
        D = collect(dD)
        @test D ≈ α .* conj.(permutedims(A, pA)) .+ β .* permutedims(B, pB) .+ γ .* C
        if can_pin
            Dh = similar(C)
            Mem.pin(Dh)
            Dh = CUDA.@sync CUTENSOR.elementwiseTrinary!(α, A, indsA, opA, β, B, indsB, opB,
                                                         γ, C, indsC, opC, Dh, indsC, opAB, opABC)
            @test Dh ≈ α .* conj.(permutedims(A, pA)) .+ β .* permutedims(B, pB) .+ γ .* C
        end

        opB = eltyB <: Complex ? CUTENSOR.CUTENSOR_OP_CONJ :
                                CUTENSOR.CUTENSOR_OP_IDENTITY
        opAB = CUTENSOR.CUTENSOR_OP_ADD
        opABC = CUTENSOR.CUTENSOR_OP_ADD
        dD = CUTENSOR.elementwiseTrinary!(α, dA, indsA, opA, β, dB, indsB, opB,
                                            γ, dC, indsC, opC, dD, indsC, opAB, opABC)
        D = collect(dD)
        @test D ≈ α .* conj.(permutedims(A, pA)) .+
                    β .* conj.(permutedims(B, pB)) .+ γ .* C
        if can_pin
            Di = similar(C)
            Mem.pin(Di)
            Di = CUDA.@sync CUTENSOR.elementwiseTrinary!(α, A, indsA, opA, β, B, indsB, opB,
                                                         γ, C, indsC, opC, Di, indsC, opAB, opABC)
            @test Di ≈ α .* conj.(permutedims(A, pA)) .+
                        β .* conj.(permutedims(B, pB)) .+ γ .* C
        end

        opA = CUTENSOR.CUTENSOR_OP_IDENTITY
        opAB = CUTENSOR.CUTENSOR_OP_MUL
        opABC = CUTENSOR.CUTENSOR_OP_ADD
        dD = CUTENSOR.elementwiseTrinary!(α, dA, indsA, opA, β, dB, indsB, opB,
                                            γ, dC, indsC, opC, dD, indsC, opAB, opABC)
        D = collect(dD)
        @test D ≈ α .* permutedims(A, pA) .* β .* conj.(permutedims(B, pB)) .+ γ .* C
        if can_pin
            Dj = similar(C)
            Mem.pin(Dj)
            Dj = CUDA.@sync CUTENSOR.elementwiseTrinary!(α, A, indsA, opA, β, B, indsB, opB,
                                                         γ, C, indsC, opC, Dj, indsC, opAB, opABC)
            @test Dj ≈ α .* permutedims(A, pA) .* β .* conj.(permutedims(B, pB)) .+ γ .* C
        end

        # test in-place, and more complicated unary and binary operations
        opA = eltyA <: Complex ? CUTENSOR.CUTENSOR_OP_IDENTITY :
                                CUTENSOR.CUTENSOR_OP_SQRT
        opB = eltyB <: Complex ? CUTENSOR.CUTENSOR_OP_IDENTITY :
                                CUTENSOR.CUTENSOR_OP_SQRT
        # because we use rand, entries of A will be positive when elty is real
        opC = eltyC <: Complex ? CUTENSOR.CUTENSOR_OP_CONJ :
                                CUTENSOR.CUTENSOR_OP_IDENTITY
        opAB = eltyD <: Complex ? CUTENSOR.CUTENSOR_OP_MUL :
                                CUTENSOR.CUTENSOR_OP_MIN
        opABC = eltyD <: Complex ? CUTENSOR.CUTENSOR_OP_ADD :
                                CUTENSOR.CUTENSOR_OP_MAX
        α = rand(eltyD)
        β = rand(eltyD)
        γ = rand(eltyD)
        dD = CUTENSOR.elementwiseTrinary!(α, dA, indsA, opA, β, dB, indsB, opB,
                                            γ, dC, indsC, opC, dC, indsC, opAB, opABC)
        D = collect(dD)
        if eltyD <: Complex
            if eltyA <: Complex && eltyB <: Complex
                @test D ≈ α .* permutedims(A, pA) .* β .* permutedims(B, pB) .+
                            γ .* conj.(C)
            elseif eltyB <: Complex
                @test D ≈ α .* sqrt.(convert.(eltyD, permutedims(A, pA))) .*
                            β .* permutedims(B, pB) .+ γ .* conj.(C)
            elseif eltyB <: Complex
                @test D ≈ α .* permutedims(A, pA) .*
                            β .* sqrt.(convert.(eltyD, permutedims(B, pB))) .+
                            γ .* conj.(C)
            else
                @test D ≈ α .* sqrt.(convert.(eltyD, permutedims(A, pA))) .*
                            β .* sqrt.(convert.(eltyD, permutedims(B, pB))) .+
                            γ .* conj.(C)
            end
        else
            @test D ≈ max.(min.(α .* sqrt.(convert.(eltyD, permutedims(A, pA))),
                                β .* sqrt.(convert.(eltyD, permutedims(B, pB)))),
                            γ .* C)
        end
    end
end

end

############################################################################################

@testset "permutations" begin

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
    @testcase for (eltyA, eltyC) in eltypes
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

end

############################################################################################

@testset "reductions" begin

eltypes = (#(Float16, Float16), #(Float16, Float32),
                        (Float32, Float32), #(Float32, Float64),
                        (Float64, Float64),
                        #(ComplexF16, ComplexF16), (ComplexF16, ComplexF32),
                        (ComplexF32, ComplexF32), #(ComplexF32, ComplexF64),
                        (ComplexF64, ComplexF64))
@testset for NA=2:5, NC = 1:NA-1
    @testcase for (eltyA, eltyC) in eltypes
        # setup
        eltyD = eltyC
        dmax = 2^div(18,NA)
        dims = rand(2:dmax, NA)
        p = randperm(NA)
        indsA = collect(('a':'z')[1:NA])
        indsC = indsA[p][1:NC]
        dimsA = dims
        dimsC = dims[p][1:NC]
        A = rand(eltyA, (dimsA...,))
        dA = CuArray(A)
        C = rand(eltyC, (dimsC...,))
        dC = CuArray(C)
        # setup
        can_pin && Mem.pin(A)

        opA = CUTENSOR.CUTENSOR_OP_IDENTITY
        opC = CUTENSOR.CUTENSOR_OP_IDENTITY
        opReduce = CUTENSOR.CUTENSOR_OP_ADD
        # simple case
        dC = CUTENSOR.reduction!(1, dA, indsA, opA, 0, dC, indsC, opC, opReduce)
        C = collect(dC)
        @test reshape(C, (dimsC..., ones(Int,NA-NC)...)) ≈
            sum(permutedims(A, p); dims = ((NC+1:NA)...,))
        if can_pin
            Csimple = zeros(eltyC, dimsC...)
            Mem.pin(Csimple)
            Csimple = CUDA.@sync CUTENSOR.reduction!(1, A, indsA, opA, 0, Csimple, indsC, opC, opReduce)
            @test reshape(Csimple, (dimsC..., ones(Int,NA-NC)...)) ≈
                sum(permutedims(A, p); dims = ((NC+1:NA)...,))
        end

        # using integers as indices
        dC = CUTENSOR.reduction!(1, dA, collect(1:NA), opA, 0, dC, p[1:NC], opC, opReduce)
        C = collect(dC)
        @test reshape(C, (dimsC..., ones(Int,NA-NC)...)) ≈
            sum(permutedims(A, p); dims = ((NC+1:NA)...,))
        if can_pin
            Cinteger = zeros(eltyC, dimsC...)
            Mem.pin(Cinteger)
            Cinteger = CUDA.@sync CUTENSOR.reduction!(1, A, collect(1:NA), opA, 0, Cinteger, p[1:NC], opC, opReduce)
            @test reshape(Cinteger, (dimsC..., ones(Int,NA-NC)...)) ≈
                sum(permutedims(A, p); dims = ((NC+1:NA)...,))
        end

        # multiplication as reduction operator
        opReduce = CUTENSOR.CUTENSOR_OP_MUL
        dC = CUTENSOR.reduction!(1, dA, indsA, opA, 0, dC, indsC, opC, opReduce)
        C = collect(dC)
        @test reshape(C, (dimsC..., ones(Int,NA-NC)...)) ≈
            prod(permutedims(A, p); dims = ((NC+1:NA)...,)) atol=eps(Float16) rtol=Base.rtoldefault(Float16)
        if can_pin
            Cmult = zeros(eltyC, dimsC...)
            Mem.pin(Cmult)
            Cmult = CUDA.@sync CUTENSOR.reduction!(1, A, indsA, opA, 0, Cmult, indsC, opC, opReduce)
            @test reshape(Cmult, (dimsC..., ones(Int,NA-NC)...)) ≈
                prod(permutedims(A, p); dims = ((NC+1:NA)...,)) atol=eps(Float16) rtol=Base.rtoldefault(Float16)
            # NOTE: this test often yields values close to 0 that do not compare approximately
        end

        # with non-trivial coefficients and conjugation
        opA = eltyA <: Complex ? CUTENSOR.CUTENSOR_OP_CONJ :
                                CUTENSOR.CUTENSOR_OP_IDENTITY
        opC = CUTENSOR.CUTENSOR_OP_IDENTITY
        opReduce = CUTENSOR.CUTENSOR_OP_ADD
        C = rand(eltyC, (dimsC...,))
        dC = CuArray(C)
        α = rand(eltyC)
        γ = rand(eltyC)
        dC = CUTENSOR.reduction!(α, dA, indsA, opA, γ, dC, indsC, opC, opReduce)
        @test reshape(collect(dC), (dimsC..., ones(Int,NA-NC)...)) ≈
            α .* conj.(sum(permutedims(A, p); dims = ((NC+1:NA)...,))) .+ γ .* C
    end
end

end

############################################################################################

end

end
end
