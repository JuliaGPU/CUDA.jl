@testset "CUTENSOR" begin

if !isdefined(CuArrays, :CUTENSOR)
@warn "Not testing CUTENSOR"
else
using CuArrays.CUTENSOR
@info "Testing CUTENSOR $(CUTENSOR.version())"

@testset "Elementwise binary" begin
    eltyA       = Float32
    eltyC       = Float32
    eltyCompute = Float32
    Ainds = ['c', 'b', 'a']
    Cinds = ['a', 'b', 'c']
    a_size = 40
    b_size = 20
    c_size = 30
    dimsA = (c_size, b_size, a_size)
    dimsC = (a_size, b_size, c_size)
    A = rand(eltyA, dimsA...)
    dA = CuArray(A)
    C = rand(eltyC, dimsC...)
    dC = CuArray(C)
    Aop = CUTENSOR.CUTENSOR_OP_IDENTITY
    Cop = CUTENSOR.CUTENSOR_OP_IDENTITY
    dD = similar(dC)
    opAC = CUTENSOR.CUTENSOR_OP_ADD

    dD = CUTENSOR.elementwiseBinary!(Float32(1.0), dA, Ainds, Aop, Float32(1.0), dC, Cinds, Cop, dD, Cinds, opAC)
    D = collect(dD)
    @test D ≈ permutedims(A, [3, 2, 1]) + C
    dD = similar(dC)
    ctA = CuTensor(dA, Ainds)
    ctC = CuTensor(dC, Cinds)
    ctD = ctA + ctC
    hD = collect(ctD.data)
    @test hD ≈ permutedims(A, [3, 2, 1]) + C
    ctD = ctA - ctC
    hD = collect(ctD.data)
    @test hD ≈ permutedims(A, [3, 2, 1]) - C
    
    ctD = LinearAlgebra.axpy!(Float32(4.), ctA, ctC)
    hD = collect(ctD.data)
    @test hD ≈ Float32(4.0)*permutedims(A, [3, 2, 1]) + C
    
    ctD = LinearAlgebra.axpby!(Float32(4.), ctA, Float32(2.), ctC)
    hD = collect(ctD.data)
    @test hD ≈ Float32(4.0)*permutedims(A, [3, 2, 1]) + Float32(2.0)*C
    
    # host memory
    #=A = rand(eltyA, dimsA...)
    C = rand(eltyC, dimsC...)
    Aop = CUTENSOR.CUTENSOR_OP_IDENTITY
    Cop = CUTENSOR.CUTENSOR_OP_IDENTITY
    opAC = CUTENSOR.CUTENSOR_OP_ADD
    D = CUTENSOR.elementwiseBinary!(Float32(1.0), A, Ainds, Aop, Float32(1.0), C, Cinds, Cop, D, Cinds, opAC)
    @test D ≈ permutedims(A, [3, 2, 1]) + C rtol=1e-6=#
end

@testset "Elementwise trinary" begin
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
end

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
    eltyA       = Float32
    eltyB       = Float32
    eltyC       = Float32
    eltyCompute = Float32
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
    dC = CUTENSOR.contraction!(one(Float32), dA, Ainds, Aop, dB, Binds, Bop, one(Float32), dC, Cinds, Cop, opOut)
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
