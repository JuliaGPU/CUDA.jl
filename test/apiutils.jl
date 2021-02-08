@testset "API utilities" begin

@testcase "@enum_without_prefix" begin
    mod = @in_module quote
        using .CUDA.APIUtils
        @enum MY_ENUM MY_ENUM_VALUE
    end
    @eval mod begin
        @enum_without_prefix MY_ENUM MY_
        @test ENUM_VALUE == MY_ENUM_VALUE
    end
end

@testcase "@checked" begin
    mod = @in_module quote
        using .CUDA.APIUtils

        const checks = Ref(0)
        macro check(ex)
            esc(quote
                $checks[] += 1
                $ex
            end)
        end
    end

    @eval mod begin
        @checked function foo()
            ccall(:jl_getpid, Cint, ())
        end

        @test checks[] == 0
        @test foo() == getpid()
        @test checks[] == 1
        @test unsafe_foo() == getpid()
        @test checks[] == 1
    end
end

@testcase "@argout" begin
    f() = 1
    f(a) = 2
    f(a,b) = 3

    @test CUDA.@argout(f()) == nothing
    @test CUDA.@argout(f(out(4))) == 4
    @test CUDA.@argout(f(out(5), out(6))) == (5,6)
end

end
