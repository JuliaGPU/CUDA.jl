@testset "version" begin

@test isa(CUDA.version(), VersionNumber)

@test isa(CUDA.release(), VersionNumber)
@test CUDA.release().patch == 0

end
