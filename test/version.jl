@testset "version" begin

@test isa(CUDAdrv.version(), VersionNumber)

end
