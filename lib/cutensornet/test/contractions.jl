n = 8
m = 16
@testset for elty in [Float32, Float64, ComplexF32, ComplexF64]
    @testset "Simple serial" begin
        modesA = ['m', 'h', 'k', 'n']
        modesB = ['u', 'k', 'h']
        modesC = ['x', 'u', 'y']
        modesD = ['m', 'x', 'n', 'y']
        extent = Dict{Char, Int}()
        extent['m'] = 8;
        extent['n'] = 8;
        extent['u'] = 8;
        extent['h'] = 8;
        extent['k'] = 8;
        extent['x'] = 8;
        extent['y'] = 8;
        extentsA = [extent[mode] for mode in modesA]
        extentsB = [extent[mode] for mode in modesB]
        extentsC = [extent[mode] for mode in modesC]
        extentsD = [extent[mode] for mode in modesD]
        A = CuArray(rand(elty, extentsA...))
        B = CuArray(rand(elty, extentsB...))
        C = CuArray(rand(elty, extentsC...))
        raw_data_in = [A, B, C]
        modes_in = [Int32.(modesA), Int32.(modesB), Int32.(modesC)]
        extents_in = [extentsA, extentsB, extentsC]
        aligns_in = UInt32.([256, 256, 256])
        aligns_out = UInt32(256)
        ctn = CuTensorNetwork(elty, modes_in, extents_in, [C_NULL, C_NULL, C_NULL], Int32[0, 0, 0], Int32.(modesD), extentsD, C_NULL)
        @testset for max_ws_size in [2^28]
            @testset for tuning in [NoAutoTune(), AutoTune()]
                ctn.input_arrs = raw_data_in
                info = rehearse_contraction(ctn, max_ws_size)
                ctn.output_arr = CUDACore.zeros(elty, extentsD...)
                ctn = perform_contraction!(ctn, info, tuning)
                @test size(ctn.output_arr) == tuple(extentsD...)
                hA = collect(A)
                hB = collect(B)
                hC = collect(C)
                # verify contraction result against CPU reference
                hD = zeros(elty, extentsD...)
                for ym in 1:extent['y'], xm in 1:extent['x'], nm in 1:extent['n'], mm in 1:extent['m']
                    s = zero(elty)
                    for hm in 1:extent['h'], km in 1:extent['k'], um in 1:extent['u']
                        s += hA[mm, hm, km, nm] * hB[um, km, hm] * hC[xm, um, ym]
                    end
                    hD[mm, xm, nm, ym] = s
                end
                D = collect(ctn.output_arr)
                @test D ≈ hD
            end
        end
    end
    @testset "QR" begin
        A = CuArray(rand(elty, n, m))
        modesA = ['n','m']
        Q = CUDACore.zeros(elty, n, n)
        R = CUDACore.zeros(elty, n, m)
        Q, R = qr!(CuTensor(A, modesA), CuTensor(Q, ['n', 'o']), CuTensor(R, ['o', 'm']))
        @test collect(Q*R) ≈ collect(A)
    end
    @testset "SVD" begin
        A = CuArray(rand(elty, n, n))
        modesA = ['n','m']
        U = CUDACore.zeros(elty, n, n)
        S = CUDACore.zeros(real(elty), n)
        V = CUDACore.zeros(elty, n, n)
        config = cuTensorNet.SVDConfig(abs_cutoff=0.0, rel_cutoff=0.0)
        U, S, V, info = svd!(CuTensor(A, modesA), CuTensor(U, ['n', 'o']), S, CuTensor(V, ['o', 'm']), svd_config=config)
        @test cuTensorNet.full_extent(info)      == n
        @test cuTensorNet.reduced_extent(info)   == n
        @test cuTensorNet.discarded_weight(info) ≈ 0.0
        @test collect(U)*diagm(collect(S))*collect(V) ≈ collect(A)
        config = cuTensorNet.CuTensorSVDConfig()
        @test cuTensorNet.abs_cutoff(config) == 0.0
        @test cuTensorNet.rel_cutoff(config) == 0.0
        @test cuTensorNet.normalization(config) == cuTensorNet.CUTENSORNET_TENSOR_SVD_NORMALIZATION_NONE
    end
    @testset "GateSplit" begin
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
        Aout, S, Bout, info = gateSplit!(CuTensor(A, modesA), CuTensor(B, modesB), CuTensor(G, modesG), CuTensor(Aout, modesAout), S, CuTensor(Bout, modesBout), svd_config=config)
        @test cuTensorNet.full_extent(info)      == z*z*2
        @test cuTensorNet.reduced_extent(info)   == z
    end
end
