@testset "profile" begin

CUDAdrv.Profile.start()
CUDAdrv.Profile.stop()

CUDAdrv.@profile begin end

end
