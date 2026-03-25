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
    @test convert(cuTensorNet.cutensornetComputeType_t, CUDACore.BFloat16) == cuTensorNet.CUTENSORNET_COMPUTE_16BF
    @test convert(Type, cuTensorNet.CUTENSORNET_COMPUTE_16BF) == CUDACore.BFloat16


    modesA = ['m', 'h', 'k', 'n']
    extent = Dict{Char, Int}()
    extent['m'] = 96;
    extent['n'] = 96;
    extent['h'] = 64;
    extent['k'] = 64;
    extentsA = [extent[mode] for mode in modesA]
    @testset for elty in [Float32, Float64, ComplexF32, ComplexF64]
        A = CuArray(rand(elty, extentsA...))
        descA = cuTensorNet.CuTensorDescriptor(A, modesA)
        @test ndims(descA) == ndims(A)
        @test size(descA)  == size(A)
        @test strides(descA) == strides(A)
    end
    # test if this constructor works
    slice_group = cuTensorNet.CuTensorNetworkSliceGroup(collect(0:63))
    @test slice_group isa cuTensorNet.CuTensorNetworkSliceGroup
end
