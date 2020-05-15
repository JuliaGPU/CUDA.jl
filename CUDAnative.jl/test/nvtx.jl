@testset "NVTX" begin


# markers

NVTX.mark("test")


# ranges

NVTX.@range "test" begin
end

NVTX.@range function test()
end
test()

@eval test2() = nothing

NVTX.@range function Main.test2(::Int)
end

NVTX.@range function Main.test2(::T) where T
end

NVTX.@range test3() = nothing

end
