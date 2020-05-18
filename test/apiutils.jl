using CUDA.APIUtils

@testset "API utilities" begin

@testset "call" begin
    # ccall throws if the lib doesn't exist, even if not called
    foo(x) = (x && ccall((:whatever, "nonexisting"), Cvoid, ()); 42)
    if VERSION < v"1.4.0-DEV.653"
        @test_throws ErrorException foo(false)
    else
        foo(false)
    end

    # @runtime_ccall prevents that
    bar(x) = (x && @runtime_ccall((:whatever, "nonexisting"), Cvoid, ()); 42)
    @test bar(false) == 42
    # but should still error nicely if actually calling the library
    @test_throws ErrorException bar(true)

    # ccall also doesn't support non-constant arguments
    lib = Ref("libjulia")
    baz() = ccall((:jl_getpid, lib[]), Cint, ())
    @test_throws TypeError baz()

    # @runtime_ccall supports that
    qux() = @runtime_ccall((:jl_getpid, lib[]), Cint, ())
    @test qux() == getpid()

    # decoding ccall/@runtime_ccall
    @test decode_ccall_function(:(ccall((:fun, :lib)))) == "fun"
    @test decode_ccall_function(:(@runtime_ccall((:fun, :lib)))) == "fun"
end

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

end
