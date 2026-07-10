# corner cases of the internal backend descriptor layer that the public graph and op
# tests cannot reach; everything else is covered end-to-end by those suites

using cuDNN:
    BackendDescriptor,
    backend_tensor,
    unsafe_destroy!,
    cudnnBackendAttributeName_t,
    cudnnDataType_t,
    CUDNN_BACKEND_TENSOR_DESCRIPTOR,
    CUDNN_DATA_FLOAT

@testset "version" begin
    @test cuDNN.version() isa VersionNumber
    @test cuDNN.version().major == cuDNN.cudnnGetProperty(CUDACore.MAJOR_VERSION)
    @test cuDNN.version().minor == cuDNN.cudnnGetProperty(CUDACore.MINOR_VERSION)
    @test cuDNN.version().patch == cuDNN.cudnnGetProperty(CUDACore.PATCH_LEVEL)
    @test cuDNN.cuda_version() isa VersionNumber
end

# destruction must be idempotent and safe to combine with finalization
d = BackendDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR)
unsafe_destroy!(d)
@test unsafe_destroy!(d) === nothing
@test Base.finalize(d) === nothing

# the attribute table derived from the enums covers every attribute
@test Set(values(cuDNN.attribute_names)) == Set(instances(cudnnBackendAttributeName_t))

# attributes round-trip through symbolic indexing
t = backend_tensor(uid=42,
                   dims=Int64[1, 2, 3, 4],
                   strides=Int64[24, 12, 4, 1],
                   dtype=CUDNN_DATA_FLOAT,
                   alignment=16)
@test t[:unique_id, Int64] == 42
@test t[:data_type, cudnnDataType_t] == CUDNN_DATA_FLOAT
@test t[:dimensions, Vector{Int64}] == Int64[1, 2, 3, 4]
@test t[:strides, Vector{Int64}] == Int64[24, 12, 4, 1]
@test_throws ArgumentError t[:bogus, Int64]
@test_throws ArgumentError t[:bogus] = 1
