using cuBLAS

@test cuBLAS.functional()

@testset "version" begin
    @test cuBLAS.version() isa VersionNumber
    @test cuBLAS.version().major == cuBLAS.cublasGetProperty(CUDACore.MAJOR_VERSION)
    @test cuBLAS.version().minor == cuBLAS.cublasGetProperty(CUDACore.MINOR_VERSION)
    @test cuBLAS.version().patch == cuBLAS.cublasGetProperty(CUDACore.PATCH_LEVEL)
end
