using cuTensorNet: CuNetwork, append_tensor!, set_output!, set_optimizer_info!, prepare_contraction!,
    contract!, autotune!, set_gradient!, set_adjoint!, compute_gradients!, attribute, attribute!,
    tensor_ids, pack, unpack!, purge_cache!, workspace_size, CuTensorNetworkContractionOptimizerInfo,
    CuTensorNetworkContractionOptimizerConfig, OptimizerConfig, CuNetworkAutotunePreference,
    NetworkAutotunePreferences, CuTensorNetworkSliceGroup, CuTensorDescriptor

function matmul_network(A::CuArray{T}, B::CuArray{T}, C::CuArray{T}; kwargs...) where {T}
    # C[i,k] = A[i,j] B[j,k], with mode labels i=1, j=2, k=3
    net = CuNetwork{T}()
    a = append_tensor!(net, A, [1, 2]; kwargs...)
    b = append_tensor!(net, B, [2, 3])
    set_output!(net, C, [1, 3])
    return net, a, b
end

@testset "contraction" begin
    @testset for elty in [Float32, Float64, ComplexF32, ComplexF64]
        A = CuArray(rand(elty, 4, 3))
        B = CuArray(rand(elty, 3, 5))
        C = CUDACore.zeros(elty, 4, 5)
        net, a, b = matmul_network(A, B, C)
        @test (a, b) == (0, 1)
        @test eltype(net) == elty
        @test_throws ArgumentError prepare_contraction!(net)
        @test_throws ArgumentError contract!(net)
        rehearse_contraction(net, 2^26)
        prepare_contraction!(net)
        @test contract!(net) === C
        CUDACore.synchronize()
        @test collect(C) ≈ collect(A) * collect(B)

        # accumulating into the output
        contract!(net; accumulate=true)
        CUDACore.synchronize()
        @test collect(C) ≈ 2 * collect(A) * collect(B)

        # a subset of slices, which here is the only one
        contract!(net; slices=CuTensorNetworkSliceGroup(0, 1, 1))
        CUDACore.synchronize()
        @test collect(C) ≈ collect(A) * collect(B)

        autotune!(net)
        autotune!(net; prefs=CuNetworkAutotunePreference(NetworkAutotunePreferences(max_iterations=1)))
        contract!(net)
        CUDACore.synchronize()
        @test collect(C) ≈ collect(A) * collect(B)
    end

    @testset "dimension mismatches" begin
        net = CuNetwork{ComplexF64}()
        @test_throws DimensionMismatch append_tensor!(net, CUDACore.zeros(ComplexF64, 2, 2), [1])
        @test_throws DimensionMismatch set_output!(net, CUDACore.zeros(ComplexF64, 2, 2), [1])
    end

    @testset "conjugation" begin
        A = CuArray(rand(ComplexF64, 3, 3))
        B = CuArray(rand(ComplexF64, 3, 3))
        C = CUDACore.zeros(ComplexF64, 3, 3)
        net, _, _ = matmul_network(A, B, C; conjugate=true)
        rehearse_contraction(net, 2^26)
        prepare_contraction!(net)
        contract!(net)
        CUDACore.synchronize()
        @test collect(C) ≈ conj(collect(A)) * collect(B)
    end
end

@testset "attributes" begin
    A = CuArray(rand(ComplexF64, 3, 3))
    B = CuArray(rand(ComplexF64, 3, 3))
    C = CUDACore.zeros(ComplexF64, 3, 3)
    net, a, b = matmul_network(A, B, C; conjugate=true, constant=true, requires_gradient=false)
    @test tensor_ids(net, cuTensorNet.CUTENSORNET_NETWORK_INPUT_TENSORS_NUM_CONJUGATED, cuTensorNet.CUTENSORNET_NETWORK_INPUT_TENSORS_CONJUGATED) == [a]
    @test tensor_ids(net, cuTensorNet.CUTENSORNET_NETWORK_INPUT_TENSORS_NUM_CONSTANT, cuTensorNet.CUTENSORNET_NETWORK_INPUT_TENSORS_CONSTANT) == [a]
    @test isempty(tensor_ids(net, cuTensorNet.CUTENSORNET_NETWORK_INPUT_TENSORS_NUM_REQUIRE_GRAD, cuTensorNet.CUTENSORNET_NETWORK_INPUT_TENSORS_REQUIRE_GRAD))
    attribute!(net, cuTensorNet.CUTENSORNET_NETWORK_COMPUTE_TYPE, cuTensorNet.CUTENSORNET_COMPUTE_64F)
    @test attribute(net, cuTensorNet.CUTENSORNET_NETWORK_COMPUTE_TYPE, cuTensorNet.cutensornetComputeType_t) == cuTensorNet.CUTENSORNET_COMPUTE_64F

    config = CuTensorNetworkContractionOptimizerConfig(OptimizerConfig(graph_num_cuts=7))
    @test attribute(config, cuTensorNet.CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_GRAPH_NUM_CUTS, Int32) == 7

    prefs = CuNetworkAutotunePreference(NetworkAutotunePreferences(max_iterations=5, intermediate_modes=1))
    @test attribute(prefs, cuTensorNet.CUTENSORNET_NETWORK_AUTOTUNE_MAX_ITERATIONS) == 5
    @test attribute(prefs, cuTensorNet.CUTENSORNET_NETWORK_AUTOTUNE_INTERMEDIATE_MODES) == 1

    desc = CuTensorDescriptor(A, [1, 2])
    @test !cuTensorNet.is_distributed(desc)
    @test cuTensorNet.local_extents(desc) == size(A)
    @test cuTensorNet.local_data_size(desc) == sizeof(A)
end

@testset "reusing a contraction path" begin
    A = CuArray(rand(ComplexF64, 3, 3))
    B = CuArray(rand(ComplexF64, 3, 3))
    C = CUDACore.zeros(ComplexF64, 3, 3)
    net, _, _ = matmul_network(A, B, C)
    info = rehearse_contraction(net, 2^26)
    data = pack(info)
    @test !isempty(data)

    # from packed data
    net2, _, _ = matmul_network(A, B, C)
    set_optimizer_info!(net2, CuTensorNetworkContractionOptimizerInfo(net2, data))
    prepare_contraction!(net2)
    contract!(net2)
    CUDACore.synchronize()
    @test collect(C) ≈ collect(A) * collect(B)

    # updating existing info
    fresh = CuTensorNetworkContractionOptimizerInfo(net2)
    @test unpack!(fresh, data) === fresh
    @test cuTensorNet.flop_count(fresh) ≈ cuTensorNet.flop_count(info)

    # the raw path can also be replaced: writing back what's read must be a no-op
    @test cuTensorNet.num_slices(fresh) == 1
    pairs = Vector{cuTensorNet.cutensornetNodePair_t}(undef, 1)
    GC.@preserve pairs begin
        path = cuTensorNet.cutensornetContractionPath_t(1, pointer(pairs))
        path_ref = Ref(path)
        cuTensorNet.cutensornetContractionOptimizerInfoGetAttribute(cuTensorNet.handle(), fresh, cuTensorNet.CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_PATH, path_ref, sizeof(path_ref))
        attribute!(fresh, cuTensorNet.CUTENSORNET_CONTRACTION_OPTIMIZER_INFO_PATH, path)
    end
    @test cuTensorNet.flop_count(fresh) ≈ cuTensorNet.flop_count(info)
end

@testset "gradients" begin
    @testset for elty in [Float64, ComplexF64]
        # c = A[i,j] B[i,j], whose gradient with respect to A is B
        A = CuArray(rand(elty, 3, 3))
        B = CuArray(rand(elty, 3, 3))
        c = CUDACore.zeros(elty, ())
        net = CuNetwork{elty}()
        a = append_tensor!(net, A, [1, 2]; requires_gradient=true)
        append_tensor!(net, B, [1, 2]; constant=true)
        set_output!(net, c, Int32[])
        @test tensor_ids(net, cuTensorNet.CUTENSORNET_NETWORK_INPUT_TENSORS_NUM_REQUIRE_GRAD, cuTensorNet.CUTENSORNET_NETWORK_INPUT_TENSORS_REQUIRE_GRAD) == [a]
        rehearse_contraction(net, 2^26)
        prepare_contraction!(net; gradients=true)

        gradient = CUDACore.zeros(elty, 3, 3)
        seed = CUDACore.fill(one(elty), ())
        @test_throws DimensionMismatch set_gradient!(net, a, CUDACore.zeros(elty, 2, 2))
        @test_throws DimensionMismatch set_adjoint!(net, CUDACore.zeros(elty, 2))
        set_gradient!(net, a, gradient)
        set_adjoint!(net, seed)
        contract!(net)
        compute_gradients!(net)
        CUDACore.synchronize()
        @test collect(c)[] ≈ sum(collect(A) .* collect(B))
        @test collect(gradient) ≈ conj(collect(B))

        compute_gradients!(net; accumulate=true)
        CUDACore.synchronize()
        @test collect(gradient) ≈ 2 * conj(collect(B))

        # a new data set requires purging the cache
        A .= 2 .* A
        @test purge_cache!(net.workspace) === net.workspace
        contract!(net)
        CUDACore.synchronize()
        @test collect(c)[] ≈ sum(collect(A) .* collect(B))
        @test workspace_size(net.workspace, cuTensorNet.CUTENSORNET_MEMSPACE_DEVICE, cuTensorNet.CUTENSORNET_WORKSPACE_SCRATCH) >= 0
    end
end

@testset "compute types" begin
    @test cuTensorNet.compute_type(Float16) == Float16
    @test_throws ArgumentError Type(cuTensorNet.cutensornetComputeType_t(0x7fffffff))
end

@testset "JULIA_DEBUG" begin
    # what makes `__init__` enable forwarding the library's logs
    withenv("JULIA_DEBUG" => "cuTensorNet") do
        @test CUDACore.isdebug(cuTensorNet)
    end
    withenv("JULIA_DEBUG" => "") do
        @test !CUDACore.isdebug(cuTensorNet)
    end
end

@testset "errors" begin
    @test cuTensorNet.error_string(cuTensorNet.CUTENSORNET_STATUS_INVALID_VALUE) != ""
    @test cuTensorNet.last_error() isa String
end
