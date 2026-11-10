using cuTensorNet: CuTensorNetwork, perform_contraction!, AutoTune, NoAutoTune, AutotunePreferences

@testset "descriptor-based contractions" begin
    @testset for elty in [Float32, ComplexF64], tuning in [NoAutoTune(), AutoTune()]
        # D[m,x,n,y] = A[m,h,k,n] B[u,k,h] C[x,u,y]
        modesA = ['m', 'h', 'k', 'n']
        modesB = ['u', 'k', 'h']
        modesC = ['x', 'u', 'y']
        modesD = ['m', 'x', 'n', 'y']
        extentsA = [4, 3, 2, 5]
        extentsB = [3, 2, 3]
        extentsC = [2, 3, 4]
        extentsD = [4, 2, 5, 4]
        A = CuArray(rand(elty, extentsA...))
        B = CuArray(rand(elty, extentsB...))
        C = CuArray(rand(elty, extentsC...))
        ctn = @test_deprecated CuTensorNetwork(elty, [Int32.(modesA), Int32.(modesB), Int32.(modesC)],
                                               [extentsA, extentsB, extentsC], [C_NULL, C_NULL, C_NULL],
                                               Int32[0, 0, 0], Int32.(modesD), extentsD, C_NULL)
        ctn.input_arrs = [A, B, C]
        ctn.output_arr = CUDACore.zeros(elty, extentsD...)
        info = rehearse_contraction(ctn, 2^28)
        @test cuTensorNet.num_slices(info) >= 1
        prefs = AutotunePreferences(max_iterations=2)
        @test (@test_deprecated perform_contraction!(ctn, info, tuning; prefs)) === ctn

        hA, hB, hC = collect(A), collect(B), collect(C)
        hD = zeros(elty, extentsD...)
        for m in 1:4, x in 1:2, n in 1:5, y in 1:4, h in 1:3, k in 1:2, u in 1:3
            hD[m, x, n, y] += hA[m, h, k, n] * hB[u, k, h] * hC[x, u, y]
        end
        @test collect(ctn.output_arr) ≈ hD
    end
end

@testset "decompositions extending LinearAlgebra" begin
    elty = ComplexF64
    A = CuArray(rand(elty, 4, 6))

    Q, R = @test_deprecated qr!(A, ['n', 'm'], CUDACore.zeros(elty, 4, 4), ['n', 'o'], CUDACore.zeros(elty, 4, 6), ['o', 'm'])
    @test collect(Q) * collect(R) ≈ collect(A)
    Q, R = @test_deprecated qr!(CuTensor(A, ['n', 'm']), CuTensor(CUDACore.zeros(elty, 4, 4), ['n', 'o']), CuTensor(CUDACore.zeros(elty, 4, 6), ['o', 'm']))
    @test collect(Q) * collect(R) ≈ collect(A)

    U, S, V, info = @test_deprecated svd!(A, ['n', 'm'], CUDACore.zeros(elty, 4, 4), ['n', 'o'], CUDACore.zeros(Float64, 4), CUDACore.zeros(elty, 4, 6), ['o', 'm'])
    @test collect(U) * diagm(collect(S)) * collect(V) ≈ collect(A)
    @test cuTensorNet.reduced_extent(info) == 4
    U, S, V, info = @test_deprecated svd!(CuTensor(A, ['n', 'm']), CuTensor(CUDACore.zeros(elty, 4, 4), ['n', 'o']), CUDACore.zeros(Float64, 4), CuTensor(CUDACore.zeros(elty, 4, 6), ['o', 'm']))
    @test collect(U) * diagm(collect(S)) * collect(V) ≈ collect(A)
end

@testset "gateSplit!" begin
    elty = ComplexF64
    A = CuArray(rand(elty, 4, 2, 4))
    B = CuArray(rand(elty, 4, 2, 4))
    G = CuArray(rand(elty, 2, 2, 2, 2))
    modesA, modesB, modesG = ['a', 'i', 'c'], ['c', 'j', 'b'], ['k', 'l', 'i', 'j']
    modesU, modesV = ['a', 'k', 'z'], ['z', 'l', 'b']
    algo = cuTensorNet.CUTENSORNET_GATE_SPLIT_ALGO_REDUCED
    _, S, _, _ = gate_split!(A, modesA, B, modesB, G, modesG, CUDACore.zeros(elty, 4, 2, 8), modesU, CUDACore.zeros(Float64, 8), CUDACore.zeros(elty, 8, 2, 4), modesV; algo)

    _, S1, _, info = @test_deprecated cuTensorNet.gateSplit!(A, modesA, B, modesB, G, modesG, CUDACore.zeros(elty, 4, 2, 8), modesU, CUDACore.zeros(Float64, 8), CUDACore.zeros(elty, 8, 2, 4), modesV; gateAlgo=algo)
    @test collect(S1) ≈ collect(S)
    @test cuTensorNet.reduced_extent(info) == 8
    _, S2, _, _ = @test_deprecated cuTensorNet.gateSplit!(CuTensor(A, modesA), CuTensor(B, modesB), CuTensor(G, modesG), CuTensor(CUDACore.zeros(elty, 4, 2, 8), modesU), CUDACore.zeros(Float64, 8), CuTensor(CUDACore.zeros(elty, 8, 2, 4), modesV))
    @test collect(S2) ≈ collect(S)
end
