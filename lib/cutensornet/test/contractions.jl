n = 8
m = 16
@testset for elty in [Float32, Float64, ComplexF32, ComplexF64]
    @testset "QR" begin
        A = CuArray(rand(elty, n, m))
        modesA = ['n','m']
        Q = CUDACore.zeros(elty, n, n)
        R = CUDACore.zeros(elty, n, m)
        Q, R = tensor_qr!(CuTensor(A, modesA), CuTensor(Q, ['n', 'o']), CuTensor(R, ['o', 'm']))
        @test collect(Q*R) ≈ collect(A)
    end
    @testset "SVD" begin
        A = CuArray(rand(elty, n, n))
        modesA = ['n','m']
        U = CUDACore.zeros(elty, n, n)
        S = CUDACore.zeros(real(elty), n)
        V = CUDACore.zeros(elty, n, n)
        config = cuTensorNet.SVDConfig(abs_cutoff=0.0, rel_cutoff=0.0)
        U, S, V, info = tensor_svd!(CuTensor(A, modesA), CuTensor(U, ['n', 'o']), S, CuTensor(V, ['o', 'm']), svd_config=config)
        @test cuTensorNet.full_extent(info)      == n
        @test cuTensorNet.reduced_extent(info)   == n
        @test cuTensorNet.discarded_weight(info) ≈ 0.0
        @test collect(U)*diagm(collect(S))*collect(V) ≈ collect(A)
        config = cuTensorNet.CuTensorSVDConfig()
        @test cuTensorNet.abs_cutoff(config) == 0.0
        @test cuTensorNet.rel_cutoff(config) == 0.0
        @test cuTensorNet.normalization(config) == cuTensorNet.CUTENSORNET_TENSOR_SVD_NORMALIZATION_NONE
    end
    @testset "gate split" begin
        a = 16
        b = 16
        c = 16
        d = 2
        f = 2
        i = 2
        j = 2
        g = 16
        h = 16
        A = CuArray(rand(elty, a, b, c, d))
        B = CuArray(rand(elty, c, f, g, h))
        G = CuArray(rand(elty, i, j, d, f))
        modesA = ['a','b','c','d']
        modesB = ['c','f','g','h']
        modesG = ['i','j','d','f']
        z = 16
        Aout = CUDACore.zeros(elty, a, b, z, i)
        modesAout = ['a','b','z','i']
        S    = CUDACore.zeros(real(elty), z)
        Bout = CUDACore.zeros(elty, z, j, g, h)
        modesBout = ['z','j','g','h']
        config = cuTensorNet.SVDConfig(abs_cutoff=0.0, rel_cutoff=0.0)
        Aout, S, Bout, info = gate_split!(CuTensor(A, modesA), CuTensor(B, modesB), CuTensor(G, modesG), CuTensor(Aout, modesAout), S, CuTensor(Bout, modesBout), svd_config=config)
        @test cuTensorNet.full_extent(info)      == z*z*2
        @test cuTensorNet.reduced_extent(info)   == z

        # the same split, contracting the gate after decomposing the inputs
        Aout2, S2, Bout2, info = gate_split!(A, modesA, B, modesB, G, modesG, CUDACore.zeros(elty, a, b, z, i), modesAout, CUDACore.zeros(real(elty), z), CUDACore.zeros(elty, z, j, g, h), modesBout;
                                             algo=cuTensorNet.CUTENSORNET_GATE_SPLIT_ALGO_REDUCED, svd_config=config)
        @test cuTensorNet.reduced_extent(info) == z
        @test collect(S2) ≈ collect(S)
    end
end
