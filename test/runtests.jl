using CUDAapi

using Test

@testset "CUDAapi" begin

@testset "library types" begin
    @test CUDAapi.PATCH_LEVEL == CUDAapi.libraryPropertyType(2)
    @test CUDAapi.C_32U == CUDAapi.cudaDataType(Complex{UInt32})
end

@testset "discovery" begin
    CUDAapi.find_binary([Sys.iswindows() ? "CHKDSK" : "true"])
    CUDAapi.find_library([Sys.iswindows() ? "NTDLL" : "c"])

    dirs = find_toolkit()
    @test !isempty(dirs)

    if haskey(ENV, "CI")
        # CI deals with plenty of CUDA versions, which makes discovery tricky.
        # dump a relevant tree of files to help debugging
        function traverse(dir, level=0)
            for entry in readdir(dir)
                print("  "^level)
                path = joinpath(dir, entry)
                if isdir(path)
                    println("└ $entry")
                    traverse(path, level+1)
                else
                    println("├ $entry")
                end
            end
        end
        for dir in dirs
            println("File tree of toolkit directory $dir:")
            traverse(dir)
        end
    end

    if VERSION < v"1.1.0-DEV.472"
        isnothing(::Any) = false
        isnothing(::Nothing) = true
    else
        isnothing = Base.isnothing
    end

    @testset "CUDA tools and libraries" begin
        @test !isnothing(find_cuda_binary("ptxas", dirs))
        ptxas = find_cuda_binary("ptxas", dirs)
        ver = parse_toolkit_version(ptxas)
        @test !isnothing(find_cuda_library("cudart", dirs, [ver]))
        if Sys.isapple() && ver.major == 10 && ver.minor == 2
            # libnvToolsExt isn't part of this release anymore?
        else
            @test !isnothing(find_cuda_library("nvtx", dirs, [v"1"]))
        end
        @test !isnothing(find_libdevice(dirs))
        @test !isnothing(find_libcudadevrt(dirs))
    end
end

@testset "availability" begin
    @test isa(has_cuda(), Bool)
    @test isa(has_cuda_gpu(), Bool)
end

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

    # FIXME: make this test work on Windows
    if !Sys.iswindows()
        # ccall also doesn't support non-constant arguments
        lib = Ref("libjulia")
        baz() = ccall((:getpid, lib[]), Cint, ())
        @test_throws TypeError baz()

        # @runtime_ccall supports that
        qux() = @runtime_ccall((:getpid, lib[]), Cint, ())
        @test qux() == getpid()
    end

    # decoding ccall/@runtime_ccall
    @test decode_ccall_function(:(ccall((:fun, :lib)))) == "fun"
    @test decode_ccall_function(:(@runtime_ccall((:fun, :lib)))) == "fun"
end

@testset "enum" begin

    @eval module Foo
    using ..CUDAapi
    @enum MY_ENUM MY_ENUM_VALUE
    @enum_without_prefix MY_ENUM MY_
    end

    @test Foo.ENUM_VALUE == Foo.MY_ENUM_VALUE

end

end
