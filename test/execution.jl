@testset "execution" begin

let
    # test outer CuDim3 constructors
    @test CUDAdrv.CuDim3((Cuint(4),Cuint(3),Cuint(2))) == CUDAdrv.CuDim3(Cuint(4),Cuint(3),Cuint(2))
    @test CUDAdrv.CuDim3((Cuint(3),Cuint(2)))          == CUDAdrv.CuDim3(Cuint(3),Cuint(2),Cuint(1))
    @test CUDAdrv.CuDim3((Cuint(2),))                  == CUDAdrv.CuDim3(Cuint(2),Cuint(1),Cuint(1))
    @test CUDAdrv.CuDim3(Cuint(2))                     == CUDAdrv.CuDim3(Cuint(2),Cuint(1),Cuint(1))

    # outer constructor should type convert
    @test CUDAdrv.CuDim3(2)       == CUDAdrv.CuDim3(Cuint(2),Cuint(1),Cuint(1))
    @test_throws InexactError CUDAdrv.CuDim3(typemax(Int64))

    # CuDim type alias should accept conveniently-typed dimensions
    @test isa(2,        CUDAdrv.CuDim)
    @test isa((2,),     CUDAdrv.CuDim)
    @test isa((2,2),    CUDAdrv.CuDim)
    @test isa((2,2,2),  CUDAdrv.CuDim)
    @test isa(Cuint(2), CUDAdrv.CuDim)
end

let
    md = CuModuleFile(joinpath(@__DIR__, "ptx/dummy.ptx"))
    dummy = CuFunction(md, "dummy")

    # different cudacall syntaxes
    cudacall(dummy, Tuple{})
    cudacall(dummy, Tuple{}; threads=1)
    cudacall(dummy, Tuple{}; threads=1, blocks=1)
    cudacall(dummy, Tuple{}; threads=1, blocks=1, shmem=0)
    cudacall(dummy, Tuple{}; threads=1, blocks=1, shmem=0, stream=CuDefaultStream())
    cudacall(dummy, ())
    cudacall(dummy, (); threads=1, blocks=1, shmem=0, stream=CuDefaultStream())

    # different launch syntaxes
    CUDAdrv.launch(dummy, 1, 1, 0, CuDefaultStream(), ())
end

let
    md = CuModuleFile(joinpath(@__DIR__, "ptx/vectorops.ptx"))
    vadd = CuFunction(md, "vadd")
    vmul = CuFunction(md, "vmul")
    vsub = CuFunction(md, "vsub")
    vdiv = CuFunction(md, "vdiv")

    a = rand(Float32, 10)
    b = rand(Float32, 10)
    ad = Mem.upload(a)
    bd = Mem.upload(b)

    # Addition
    let
        c = zeros(Float32, 10)
        cd = Mem.alloc(c)
        cudacall(vadd,
                 (CuPtr{Cfloat},CuPtr{Cfloat},CuPtr{Cfloat}), ad, bd, cd;
                 threads=10)
        Mem.download!(c, cd)
        @test c ≈ a+b
    end

    # Subtraction
    let
        c = zeros(Float32, 10)
        cd = Mem.alloc(c)
        cudacall(vsub,
                 (CuPtr{Cfloat},CuPtr{Cfloat},CuPtr{Cfloat}), ad, bd, cd;
                 threads=10)
        Mem.download!(c, cd)
        @test c ≈ a-b
    end

    # Multiplication
    let
        c = zeros(Float32, 10)
        cd = Mem.alloc(c)
        cudacall(vmul,
                 (CuPtr{Cfloat},CuPtr{Cfloat},CuPtr{Cfloat}), ad, bd, cd;
                 threads=10)
        Mem.download!(c, cd)
        @test c ≈ a.*b
    end

    # Division
    let
        c = zeros(Float32, 10)
        cd = Mem.alloc(c)
        cudacall(vdiv,
                 (CuPtr{Cfloat},CuPtr{Cfloat},CuPtr{Cfloat}), ad, bd, cd;
                 threads=10)
        Mem.download!(c, cd)
        @test c ≈ a./b
    end
end

@testset "attributes" begin

md = CuModuleFile(joinpath(@__DIR__, "ptx/dummy.ptx"))
dummy = CuFunction(md, "dummy")

val = attributes(dummy)[CUDAdrv.FUNC_ATTRIBUTE_SHARED_SIZE_BYTES]

if CUDAdrv.version() >= v"9.0"
    attributes(dummy)[CUDAdrv.FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES] = val
end

end

end
