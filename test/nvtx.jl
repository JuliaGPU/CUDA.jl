@testcase "NVTX" begin

@testcase "markers" begin
    NVTX.mark("test")
end

@testcase "ranges" begin
    NVTX.@range "test" begin
    end

    NVTX.@range function test()
    end
    test()

    outer = @in_module Expr(:module, true, :Inner, quote end)
    # NOTE: `quote module end` doesn't work
    @eval outer.Inner begin
        test2() = nothing
    end

    @eval outer begin
        call_external_dummy() = @cuda external_dummy()

        NVTX.@range function Inner.test2(::Int)
        end

        NVTX.@range function Inner.test2(::T) where T
        end
    end

    NVTX.@range test3() = nothing
end

end
