using Test

using CUDA
@info "CUDA information:\n" * sprint(io->CUDA.versioninfo(io))

using LinearAlgebra
using cuTENSOR
using cuTensorNet
@test cuTENSOR.has_cutensor()
@test cuTensorNet.has_cutensornet()
@info "cuTensorNet version: $(cuTensorNet.version()) (built for CUDA $(cuTensorNet.cuda_version()))"

import cuTensorNet: CuTensorNetwork, rehearse_contraction, perform_contraction!, gateSplit!, AutoTune, NoAutoTune

using TensorOperations

@testset "cuTensorNet" begin
    @testset "Helpers and types" begin
        @test convert(cuTensorNet.cutensornetComputeType_t, Int8)  == cuTensorNet.CUTENSORNET_COMPUTE_8I
        @test convert(cuTensorNet.cutensornetComputeType_t, UInt8) == cuTensorNet.CUTENSORNET_COMPUTE_8U
        @test convert(cuTensorNet.cutensornetComputeType_t, Float16) == cuTensorNet.CUTENSORNET_COMPUTE_16F
        @test convert(cuTensorNet.cutensornetComputeType_t, Int32)  == cuTensorNet.CUTENSORNET_COMPUTE_32I
        @test convert(cuTensorNet.cutensornetComputeType_t, UInt32) == cuTensorNet.CUTENSORNET_COMPUTE_32U
        @test_throws ArgumentError("cuTensorNet type equivalent for compute type ComplexF64 does not exist!") convert(cuTensorNet.cutensornetComputeType_t, ComplexF64)
        @test convert(Type, cuTensorNet.CUTENSORNET_COMPUTE_8I) == Int8
        @test convert(Type, cuTensorNet.CUTENSORNET_COMPUTE_8U) == UInt8
        @test convert(Type, cuTensorNet.CUTENSORNET_COMPUTE_16F) == Float16
        @test convert(Type, cuTensorNet.CUTENSORNET_COMPUTE_32F) == Float32
        @test convert(Type, cuTensorNet.CUTENSORNET_COMPUTE_32U) == UInt32
        @test convert(Type, cuTensorNet.CUTENSORNET_COMPUTE_32I) == Int32
        @test convert(Type, cuTensorNet.CUTENSORNET_COMPUTE_64F) == Float64

        
        modesA = ['m', 'h', 'k', 'n']
        extent = Dict{Char, Int}()
        extent['m'] = 96;
        extent['n'] = 96;
        extent['h'] = 64;
        extent['k'] = 64;
        extentsA = [extent[mode] for mode in modesA]
        @testset for elty in [Float32, Float64, ComplexF32, ComplexF64]
            A = CUDA.rand(elty, extentsA...)
            descA = cuTensorNet.CuTensorDescriptor(A, modesA)
            @test ndims(descA) == ndims(A)
            @test size(descA)  == size(A)
            @test strides(descA) == strides(A)
        end
        # test if this constructor works
        slice_group = cuTensorNet.CuTensorNetworkSliceGroup(collect(0:63))
        @test slice_group isa cuTensorNet.CuTensorNetworkSliceGroup
    end
    @testset "Errors" begin
        @test sprint(showerror, cuTensorNet.CUTENSORNETError(cuTensorNet.CUTENSORNET_STATUS_SUCCESS)) == "CUTENSORNETError: the operation completed successfully (code 0, CUTENSORNET_STATUS_SUCCESS)"
        @test cuTensorNet.description(cuTensorNet.CUTENSORNETError(cuTensorNet.CUTENSORNET_STATUS_NOT_INITIALIZED)) == "the library was not initialized"
        @test cuTensorNet.description(cuTensorNet.CUTENSORNETError(cuTensorNet.CUTENSORNET_STATUS_ALLOC_FAILED)) == "the resource allocation failed"
        @test cuTensorNet.description(cuTensorNet.CUTENSORNETError(cuTensorNet.CUTENSORNET_STATUS_INVALID_VALUE)) == "an invalid value was used as an argument"
        @test cuTensorNet.description(cuTensorNet.CUTENSORNETError(cuTensorNet.CUTENSORNET_STATUS_ARCH_MISMATCH)) == "an absent device architectural feature is required"
        @test cuTensorNet.description(cuTensorNet.CUTENSORNETError(cuTensorNet.CUTENSORNET_STATUS_EXECUTION_FAILED)) == "the GPU program failed to execute"
        @test cuTensorNet.description(cuTensorNet.CUTENSORNETError(cuTensorNet.CUTENSORNET_STATUS_INTERNAL_ERROR)) == "an internal operation failed"
        @test cuTensorNet.description(cuTensorNet.CUTENSORNETError(cuTensorNet.CUTENSORNET_STATUS_NOT_SUPPORTED)) == "the API is not supported by the backend."
        @test cuTensorNet.description(cuTensorNet.CUTENSORNETError(cuTensorNet.CUTENSORNET_STATUS_LICENSE_ERROR)) == "error checking current licensing."
        @test cuTensorNet.description(cuTensorNet.CUTENSORNETError(cuTensorNet.CUTENSORNET_STATUS_CUBLAS_ERROR)) == "a call to CUBLAS did not succeed."
        @test cuTensorNet.description(cuTensorNet.CUTENSORNETError(cuTensorNet.CUTENSORNET_STATUS_CUDA_ERROR)) == "some unknown CUDA error has occurred."
        @test cuTensorNet.description(cuTensorNet.CUTENSORNETError(cuTensorNet.CUTENSORNET_STATUS_INSUFFICIENT_WORKSPACE)) == "the workspace on the device is too small to execute."
        @test cuTensorNet.description(cuTensorNet.CUTENSORNETError(cuTensorNet.CUTENSORNET_STATUS_INSUFFICIENT_DRIVER)) == "the driver version is insufficient."
        @test cuTensorNet.description(cuTensorNet.CUTENSORNETError(cuTensorNet.CUTENSORNET_STATUS_IO_ERROR)) == "an error occurred related to file IO."
        @test cuTensorNet.description(cuTensorNet.CUTENSORNETError(cuTensorNet.CUTENSORNET_STATUS_CUTENSOR_VERSION_MISMATCH)) == "the dynamically linked cuTENSOR library is incompatible."
        @test cuTensorNet.description(cuTensorNet.CUTENSORNETError(cuTensorNet.CUTENSORNET_STATUS_NO_DEVICE_ALLOCATOR)) == "drawing device memory from a mempool is requested, but the mempool is not set."
        @test cuTensorNet.description(cuTensorNet.CUTENSORNETError(cuTensorNet.CUTENSORNET_STATUS_ALL_HYPER_SAMPLES_FAILED)) == "all hyper samples failed for one or more errors."
    end
    n = 8
    m = 16
    k = 32
    @testset for elty in [Float32, Float64, ComplexF32, ComplexF64]
        @testset "Simple serial" begin
            modesA = ['m', 'h', 'k', 'n']
            modesB = ['u', 'k', 'h']
            modesC = ['x', 'u', 'y']
            modesD = ['m', 'x', 'n', 'y']
            extent = Dict{Char, Int}()
            extent['m'] = 96;
            extent['n'] = 96;
            extent['u'] = 96;
            extent['h'] = 64;
            extent['k'] = 64;
            extent['x'] = 64;
            extent['y'] = 64;
            extentsA = [extent[mode] for mode in modesA]
            extentsB = [extent[mode] for mode in modesB]
            extentsC = [extent[mode] for mode in modesC]
            extentsD = [extent[mode] for mode in modesD]
            A = CUDA.rand(elty, extentsA...)
            B = CUDA.rand(elty, extentsB...)
            C = CUDA.rand(elty, extentsC...)
            raw_data_in = [A, B, C]
            modes_in = [Int32.(modesA), Int32.(modesB), Int32.(modesC)]
            extents_in = [extentsA, extentsB, extentsC]
            aligns_in = UInt32.([256, 256, 256])
            aligns_out = UInt32(256)
            ctn = CuTensorNetwork(elty, modes_in, extents_in, [C_NULL, C_NULL, C_NULL], Int32[0, 0, 0], Int32.(modesD), extentsD, C_NULL)
            @testset for max_ws_size in [2^28, 2^32]
                @testset for tuning in [NoAutoTune(), AutoTune()]
                    ctn.input_arrs = raw_data_in
                    info = rehearse_contraction(ctn, max_ws_size)
                    ctn.output_arr = CUDA.zeros(elty, extentsD...)
                    ctn = perform_contraction!(ctn, info, tuning)
                    @test size(ctn.output_arr) == tuple(extentsD...)
                    hA = collect(A)
                    hB = collect(B)
                    hC = collect(C)
                    hD = zeros(elty, extentsD...)
                    @tensor begin
                        hD[m, x, n, y] := hA[m, h, k, n] * hB[u, k, h] * hC[x, u, y]
                    end
                    D = collect(ctn.output_arr)
                    @test D ≈ hD
                end
            end
        end
        @testset "QR" begin
            A = CUDA.rand(elty, n, m)
            modesA = ['n','m']
            Q = CUDA.zeros(elty, n, n)
            R = CUDA.zeros(elty, n, m)
            Q, R = qr!(CuTensor(A, modesA), CuTensor(Q, ['n', 'o']), CuTensor(R, ['o', 'm']))
            @test collect(Q*R) ≈ collect(A)
        end
        @testset "SVD" begin
            A = CUDA.rand(elty, n, n)
            modesA = ['n','m']
            U = CUDA.zeros(elty, n, n)
            S = CUDA.zeros(real(elty), n)
            V = CUDA.zeros(elty, n, n)
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
            A = CUDA.rand(elty, a, b, c, d)
            B = CUDA.rand(elty, c, f, g, h)
            G = CUDA.rand(elty, i, j, d, f)
            modesA = ['a','b','c','d']
            modesB = ['c','f','g','h']
            modesG = ['i','j','d','f']
            z = 16
            Aout = CUDA.zeros(elty, a, b, z, i)
            modesAout = ['a','b','z','i']
            S    = CUDA.zeros(real(elty), z)
            Bout = CUDA.zeros(elty, z, j, g, h)
            modesBout = ['z','j','g','h']
            config = cuTensorNet.SVDConfig(abs_cutoff=0.0, rel_cutoff=0.0)
            Aout, S, Bout, info = gateSplit!(CuTensor(A, modesA), CuTensor(B, modesB), CuTensor(G, modesG), CuTensor(Aout, modesAout), S, CuTensor(Bout, modesBout), svd_config=config)
            @test cuTensorNet.full_extent(info)      == z*z*2
            @test cuTensorNet.reduced_extent(info)   == z
        end
    end
end
