@testset "NVTX" begin


# markers

NVTX.mark("test")


# ranges

NVTX.@range "test" begin
end

NVTX.@range function test()
end
test()


end
