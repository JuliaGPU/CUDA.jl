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
    ptx = readstring(f)
    close(f)

    md = CuModule(ptx)
    vadd = CuFunction(md, "vadd")

    md2 = CuModuleFile(joinpath(@__DIR__, "ptx/vadd.ptx"))
    @test md != md2
end

@test_throws_cuerror CUDAdrv.ERROR_INVALID_IMAGE CuModule("foobar")


@testset "globals" begin
    md = CuModuleFile(joinpath(@__DIR__, "ptx/global.ptx"))

    var = CuGlobal{Int32}(md, "foobar")
    @test eltype(var) == Int32
    @test eltype(typeof(var)) == Int32

    @test_throws ArgumentError CuGlobal{Int64}(md, "foobar")

    set(var, Int32(42))
    @test get(var) == Int32(42)
end


@testset "linker" begin
    link = CuLink()
    @test link == link
    @test link != CuLink()

    # regular string
    open(joinpath(@__DIR__, "ptx/empty.ptx")) do f
        addData(link, "vadd_parent", readstring(f), CUDAdrv.PTX)
    end

    # string as vector of bytes
    open(joinpath(@__DIR__, "ptx/empty.ptx")) do f
        addData(link, "vadd_parent", convert(Vector{UInt8}, readstring(f)), CUDAdrv.PTX)
    end

    # PTX code containing \0
    @test_throws ArgumentError addData(link, "vadd_parent", "\0", CUDAdrv.PTX)
    @test_throws ArgumentError addData(link, "vadd_parent", convert(Vector{UInt8}, "\0"), CUDAdrv.PTX)

    # object data containing \0
    # NOTE: apparently, on Windows cuLinkAddData _does_ accept object data containing \0
    if !is_windows()
        @test_throws_cuerror CUDAdrv.ERROR_UNKNOWN addData(link, "vadd_parent", "\0", CUDAdrv.OBJECT)
        @test_throws_cuerror CUDAdrv.ERROR_UNKNOWN addData(link, "vadd_parent", convert(Vector{UInt8}, "\0"), CUDAdrv.OBJECT)
    end
end

let
    link = CuLink()
    addFile(link, joinpath(@__DIR__, "ptx/vadd_child.ptx"), CUDAdrv.PTX)
    open(joinpath(@__DIR__, "ptx/vadd_parent.ptx")) do f
        addData(link, "vadd_parent", readstring(f), CUDAdrv.PTX)
    end

    obj = complete(link)
    md = CuModule(obj)

    vadd = CuFunction(md, "vadd")
end

end
