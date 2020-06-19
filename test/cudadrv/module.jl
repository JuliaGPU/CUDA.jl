@testset "module" begin

let
    md = CuModuleFile(joinpath(@__DIR__, "ptx/vadd.ptx"))

    vadd = CuFunction(md, "vadd")
end

# comparisons
let
    md = CuModuleFile(joinpath(@__DIR__, "ptx/vectorops.ptx"))
    vadd = CuFunction(md, "vadd")
    @test vadd == vadd
    @test vadd != CuFunction(md, "vdiv")
end

let
    f = open(joinpath(@__DIR__, "ptx/vadd.ptx"))
    ptx = read(f, String)
    close(f)

    md = CuModule(ptx)
    vadd = CuFunction(md, "vadd")

    md2 = CuModuleFile(joinpath(@__DIR__, "ptx/vadd.ptx"))
    @test md != md2
end

memcheck || @test_throws_cuerror CUDA.ERROR_INVALID_IMAGE CuModule("foobar")


@testset "globals" begin
    md = CuModuleFile(joinpath(@__DIR__, "ptx/global.ptx"))

    var = CuGlobal{Int32}(md, "foobar")
    @test eltype(var) == Int32
    @test eltype(typeof(var)) == Int32

    @test_throws ArgumentError CuGlobal{Int64}(md, "foobar")

    var[] = Int32(42)
    @test var[] == Int32(42)
end


@testset "linker" begin
    link = CuLink()
    @test link == link
    @test link != CuLink()

    # PTX code
    open(joinpath(@__DIR__, "ptx/empty.ptx")) do f
        add_data!(link, "vadd_parent", read(f, String))
    end
    @test_throws ArgumentError add_data!(link, "vadd_parent", "\0")

    # object code
    # TODO: test with valid object code
    # NOTE: apparently, on Windows cuLinkAddData! _does_ accept object data containing \0
    if !Sys.iswindows()
        memcheck || @test_throws_cuerror CUDA.ERROR_UNKNOWN add_data!(link, "vadd_parent", UInt8[0])
    end
end

let
    link = CuLink()
    add_file!(link, joinpath(@__DIR__, "ptx/vadd_child.ptx"), CUDA.JIT_INPUT_PTX)
    open(joinpath(@__DIR__, "ptx/vadd_parent.ptx")) do f
        add_data!(link, "vadd_parent", read(f, String))
    end

    obj = complete(link)
    md = CuModule(obj)

    vadd = CuFunction(md, "vadd")

    options = Dict{CUDA.CUjit_option,Any}()
    options[CUDA.JIT_GENERATE_LINE_INFO] = true

    md = CuModule(obj, options)
    vadd = CuFunction(md, "vadd")
end

end
