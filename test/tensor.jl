@testset "CUTENSOR" begin

if !isdefined(CuArrays, :CUTENSOR)
@warn "Not testing CUTENSOR"
else
using CuArrays.CUTENSOR
@info "Testing CUTENSOR"

@testset "Elementwise binary" begin
    eltypes = (Float16, Float32, Float64, # ComplexF16,
                ComplexF32, ComplexF64)
    @testset for N=2:5
        # @testset for (eltyA, eltyC) in Base.Iterators.product(eltypes, eltypes)
        # mixed precision does not seem to work currently
        @testset for elty in eltypes
            eltyA = elty
            eltyC = elty
            eltyD = promote_type(eltyA, eltyC)

            dmax = 2^div(16,N)
            dims = rand(2:dmax, N)
            p = randperm(N)
            Ainds = collect(('a':'z')[1:N])
            Cinds = Ainds[p]
            dimsA = dims
            dimsC = dims[p]
            A = rand(eltyA, dimsA...)
            dA = CuArray(A)
            C = rand(eltyC, dimsC...)
            dC = CuArray(C)
            # simple
            Aop = CUTENSOR.CUTENSOR_OP_IDENTITY
            Cop = CUTENSOR.CUTENSOR_OP_IDENTITY
            dD = similar(dC, eltyD)
            opAC = CUTENSOR.CUTENSOR_OP_ADD
            dD = CUTENSOR.elementwiseBinary!(one(eltyA), dA, Ainds, Aop, one(eltyC), dC, Cinds, Cop, dD, Cinds, opAC)
            D = collect(dD)
            @test D ≈ permutedims(A, p) .+ C
            # using integers as indices
            dD = CUTENSOR.elementwiseBinary!(one(eltyA), dA, collect(1:N), Aop, one(eltyC), dC, p, Cop, dD, p, opAC)
            D = collect(dD)
            @test D ≈ permutedims(A, p) .+ C
            # multiplication as binary operator
            opAC = CUTENSOR.CUTENSOR_OP_MUL
            dD = CUTENSOR.elementwiseBinary!(one(eltyA), dA, Ainds, Aop, one(eltyC), dC, Cinds, Cop, dD, Cinds, opAC)
            D = collect(dD)
            @test D ≈ permutedims(A, p) .* C
            # with non-trivial coefficients
            Aop = eltyA <: Complex ? CUTENSOR.CUTENSOR_OP_CONJ :
                                    CUTENSOR.CUTENSOR_OP_IDENTITY
            Cop = CUTENSOR.CUTENSOR_OP_IDENTITY
            opAC = CUTENSOR.CUTENSOR_OP_ADD
            α = rand(eltyA)
            β = rand(eltyC)
            dD = CUTENSOR.elementwiseBinary!(α, dA, Ainds, Aop, β, dC, Cinds, Cop, dD, Cinds, opAC)
            D = collect(dD)
            @test D ≈ α.*conj.(permutedims(A, p)) .+ β.*C
            # test in-place, and more complicated unary and binary operations
            if eltyD == eltyC
                Aop = eltyA <: Complex ? CUTENSOR.CUTENSOR_OP_IDENTITY :
                                        CUTENSOR.CUTENSOR_OP_SQRT
                # because we use rand, entries of A will be positive when elty is real
                Cop = eltyC <: Complex ? CUTENSOR.CUTENSOR_OP_CONJ :
                                        CUTENSOR.CUTENSOR_OP_IDENTITY
                opAC = eltyD <: Complex ? CUTENSOR.CUTENSOR_OP_ADD :
                                        CUTENSOR.CUTENSOR_OP_MAX
                α = rand(eltyA)
                β = rand(eltyC)
                dD = CUTENSOR.elementwiseBinary!(α, dA, Ainds, Aop, β, dC, Cinds, Cop, dC, Cinds, opAC)
                D = collect(dC)
                if eltyD <: Complex
                    if eltyA <: Complex
                        @test D ≈ α.*permutedims(A, p) .+ β.*conj.(C)
                    else
                        @test D ≈ α.*sqrt.(permutedims(A, p)) .+ β.*conj.(C)
                    end
                else
                    @test D ≈ max.(α.*sqrt.(permutedims(A, p)), β.*C)
                end
            end

            # using CuTensor type
            dA = CuArray(A)
            dC = CuArray(C)
            ctA = CuTensor(dA, Ainds)
            ctC = CuTensor(dC, Cinds)
            ctD = ctA + ctC
            hD = collect(ctD.data)
            @test hD ≈ permutedims(A, p) .+ C
            ctD = ctA - ctC
            hD = collect(ctD.data)
            @test hD ≈ permutedims(A, p) .- C

            α = rand(eltyA)
            ctD = LinearAlgebra.axpy!(α, ctA, ctC)
            hD = collect(ctD.data)
            @test hD ≈ α.*permutedims(A, p) .+ C

            β = rand(eltyA)
            ctD = LinearAlgebra.axpby!(α, ctA, β, ctC)
            hD = collect(ctD.data)
            @test hD ≈ α.*permutedims(A, p) .+ β.*C
        end
    end

#=@testset "Elementwise trinary" begin
    eltyA       = Float32
    eltyB       = Float32
    eltyC       = Float32
    eltyCompute = Float32
    Ainds = ['c', 'b', 'a']
    Binds = ['c', 'a', 'b']
    Cinds = ['a', 'b', 'c']
    a_size = 30
    b_size = 20
    c_size = 40
    dimsA = (c_size, b_size, a_size)
    dimsB = (c_size, a_size, b_size)
    dimsC = (a_size, b_size, c_size)
    A = rand(eltyA, dimsA...)
    dA = CuArray(A)
    B = rand(eltyB, dimsB...)
    dB = CuArray(B)
    C = rand(eltyC, dimsC...)
    dC = CuArray(C)
    Aop = CUTENSOR.CUTENSOR_OP_IDENTITY
    Bop = CUTENSOR.CUTENSOR_OP_IDENTITY
    Cop = CUTENSOR.CUTENSOR_OP_IDENTITY
    dD = similar(dC)
    opAB = CUTENSOR.CUTENSOR_OP_ADD
    opABC = CUTENSOR.CUTENSOR_OP_ADD
    dD = CUTENSOR.elementwiseTrinary!(Float32(1.0), dA, Ainds, Aop, Float32(1.0), dB, Binds, Bop, Float32(1.0), dC, Cinds, Cop, dD, Cinds, opAB, opABC)
    D = collect(dD)
    @test D ≈ permutedims(A, [3, 2, 1]) + permutedims(B, [2, 3, 1]) + C

    #=A = rand(eltyA, dimsA...)
    B = rand(eltyB, dimsB...)
    C = rand(eltyC, dimsC...)
    Aop = CUTENSOR.CUTENSOR_OP_IDENTITY
    Bop = CUTENSOR.CUTENSOR_OP_IDENTITY
    Cop = CUTENSOR.CUTENSOR_OP_IDENTITY
    opAB = CUTENSOR.CUTENSOR_OP_ADD
    opABC = CUTENSOR.CUTENSOR_OP_ADD
    D = CUTENSOR.elementwiseTrinary!(Float32(1.0), A, Ainds, Aop, Float32(1.0), B, Binds, Bop, Float32(1.0), C, Cinds, Cop, D, Cinds, opAB, opABC)
    @test D ≈ permutedims(A, [3, 2, 1]) + permutedims(B, [2, 3, 1]) + C=#
end=#

@testset "Permutations" begin
    eltyA       = Float32
    eltyC       = Float32
    eltyCompute = Float32
    Ainds = ['w', 'h', 'c', 'n']
    Cinds = ['c', 'w', 'h', 'n']
    small_size = 32
    big_size = 64
    dimsA = (small_size, big_size, big_size, big_size)
    dimsC = (big_size, small_size, big_size, big_size)
    A = rand(eltyA, dimsA...)
    dA = CuArray(A)
    dC = cuzeros(eltyC, dimsC...)
    dC = CUTENSOR.permutation!(Float32(1.), dA, Ainds, dC, Cinds)
    C  = collect(dC)
    hC = permutedims(A, (3, 1, 2, 4))
    @test hC == C

    #=A = rand(eltyA, dimsA...)
    C = CUTENSOR.permutation!(Float32(1.), A, Ainds, C, Cinds)
    hC = permutedims(A, (3, 1, 2, 4))
    @test hC == C=#
end

@testset "Contraction" begin
    @testset for (eltyA, eltyB, eltyC, eltyCompute) in ((Float32, Float32, Float32, Float32),
                                                        (Float16, Float16, Float16, Float16),
                                                        (Float64, Float64, Float64, Float64),
                                                        (ComplexF32, ComplexF32, ComplexF32, ComplexF32),
                                                        (ComplexF64, ComplexF64, ComplexF64, ComplexF64))
        Ainds = ['m', 'k']
        Binds = ['k', 'n']
        Cinds = ['m', 'n']
        m_size = 32
        k_size = 16
        n_size = 32
        dimsA = (m_size, k_size)
        dimsB = (k_size, n_size)
        dimsC = (m_size, n_size)
        A = rand(eltyA, dimsA...)
        dA = CuArray(A)
        B = rand(eltyB, dimsB...)
        dB = CuArray(B)
        C = zeros(eltyC, dimsC...)
        dC = CuArray(C)
        Aop = CUTENSOR.CUTENSOR_OP_IDENTITY
        Bop = CUTENSOR.CUTENSOR_OP_IDENTITY
        Cop = CUTENSOR.CUTENSOR_OP_IDENTITY
        opOut = CUTENSOR.CUTENSOR_OP_IDENTITY
        dC = CUTENSOR.contraction!(one(eltyA), dA, Ainds, Aop, dB, Binds, Bop, one(eltyC), dC, Cinds, Cop, opOut)
        C = collect(dC)
        @test C ≈ A * B

        ctA = CuTensor(dA, Ainds)
        ctB = CuTensor(dB, Binds)
        ctC = CuTensor(dC, Cinds)
        dC = LinearAlgebra.mul!(ctC, ctA, ctB)
        C, Cinds = collect(dC)
        @test C ≈ A * B

        dC = ctA * ctB
        C, Cinds = collect(dC)
        @test C ≈ A * B

        #=A = rand(eltyA, dimsA...)
        B = rand(eltyB, dimsB...)
        C = zeros(eltyC, dimsC...)
        Aop = CUTENSOR.CUTENSOR_OP_IDENTITY
        Bop = CUTENSOR.CUTENSOR_OP_IDENTITY
        Cop = CUTENSOR.CUTENSOR_OP_IDENTITY
        opOut = CUTENSOR.CUTENSOR_OP_IDENTITY
        C = CUTENSOR.contraction!(one(Float32), A, Ainds, Aop, B, Binds, Bop, one(Float32), C, Cinds, Cop, opOut)
        @test C ≈ A * B=#
    end
end

end

end
