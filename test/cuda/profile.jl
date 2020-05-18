@testset "profile" begin

CUDA.Profile.start()
CUDA.Profile.stop()

CUDA.@profile begin end

end
