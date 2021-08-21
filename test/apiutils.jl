using CUDA.APIUtils

@testset "@enum_without_prefix" begin
    mod = @eval module $(gensym())
        using CUDA.APIUtils
        @enum MY_ENUM MY_ENUM_VALUE
        @enum_without_prefix MY_ENUM MY_
    end

    @test mod.ENUM_VALUE == mod.MY_ENUM_VALUE
end

@testset "@checked" begin
    mod = @eval module $(gensym())
        using CUDA.APIUtils

        const checks = Ref(0)
        macro check(ex)
            esc(quote
                $checks[] += 1
                $ex
            end)
        end

        @checked function foo()
            ccall(:jl_getpid, Cint, ())
        end
    end

    @test mod.checks[] == 0
    @test mod.foo() == getpid()
    @test mod.checks[] == 1
    @test mod.unsafe_foo() == getpid()
    @test mod.checks[] == 1
end
