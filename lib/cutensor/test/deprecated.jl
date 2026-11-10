using cuTENSOR
using cuTENSOR: OP_IDENTITY, OP_ADD

@testset "deprecated names" begin
    A = rand(Float32, 4, 3)
    B = rand(Float32, 3, 4)
    C = rand(Float32, 3, 4)
    dA, dB, dC = CuArray(A), CuArray(B), CuArray(C)

    dD = similar(dC)
    @test_deprecated cuTENSOR.elementwise_binary_execute!(1, dA, ['i', 'j'], OP_IDENTITY,
                                                          2, dC, ['j', 'i'], OP_IDENTITY,
                                                          dD, ['j', 'i'], OP_ADD)
    @test collect(dD) ≈ permutedims(A) + 2C

    plan = cuTENSOR.plan_elementwise_binary(dA, ['i', 'j'], OP_IDENTITY,
                                            dC, ['j', 'i'], OP_IDENTITY,
                                            dD, ['j', 'i'], OP_ADD)
    dD = similar(dC)
    @test_deprecated cuTENSOR.elementwise_binary_execute!(plan, 1, dA, 2, dC, dD)
    @test collect(dD) ≈ permutedims(A) + 2C
    CUDACore.unsafe_free!(plan)

    dD = similar(dC)
    @test_deprecated cuTENSOR.elementwise_trinary_execute!(1, dA, ['i', 'j'], OP_IDENTITY,
                                                           2, dB, ['j', 'i'], OP_IDENTITY,
                                                           3, dC, ['j', 'i'], OP_IDENTITY,
                                                           dD, ['j', 'i'], OP_ADD, OP_ADD)
    @test collect(dD) ≈ permutedims(A) + 2B + 3C

    plan = cuTENSOR.plan_elementwise_trinary(dA, ['i', 'j'], OP_IDENTITY,
                                             dB, ['j', 'i'], OP_IDENTITY,
                                             dC, ['j', 'i'], OP_IDENTITY,
                                             dD, ['j', 'i'], OP_ADD, OP_ADD)
    dD = similar(dC)
    @test_deprecated cuTENSOR.elementwise_trinary_execute!(plan, 1, dA, 2, dB, 3, dC, dD)
    @test collect(dD) ≈ permutedims(A) + 2B + 3C
    CUDACore.unsafe_free!(plan)
end
