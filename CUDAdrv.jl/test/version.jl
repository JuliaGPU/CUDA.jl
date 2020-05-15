@testset "version" begin

@test isa(CUDAdrv.version(), VersionNumber)

@test isa(CUDAdrv.release(), VersionNumber)
@test CUDAdrv.release().patch == 0

end
