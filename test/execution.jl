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
    cudacall(dummy, 1, 1, ())
    cudacall(dummy, 1, 1, 0, CuDefaultStream(), ())
    cudacall(dummy, 1, 1, (); shmem=0, stream=CuDefaultStream())
    cudacall(dummy, 1, 1, Tuple{})
    cudacall(dummy, 1, 1, 0, CuDefaultStream(), Tuple{})
    cudacall(dummy, 1, 1, Tuple{}; shmem=0, stream=CuDefaultStream())
    ## this one is wrong, but used to trigger an overflow
    @test_throws MethodError cudacall(dummy, 1, 1, CuDefaultStream(), 0, Tuple{})
    ## bug in NTuple usage
    cudacall(dummy, 1, 1, 0, CuDefaultStream(), Tuple{Tuple{Int64},Int64}, (1,), 1)
end

let
    md = CuModuleFile(joinpath(@__DIR__, "ptx/vectorops.ptx"))
    vadd = CuFunction(md, "vadd")
    vmul = CuFunction(md, "vmul")
    vsub = CuFunction(md, "vsub")
    vdiv = CuFunction(md, "vdiv")

    a = rand(Float32, 10)
    b = rand(Float32, 10)
    ad = CuArray(a)
    bd = CuArray(b)

    # Addition
    let
        c = zeros(Float32, 10)
        cd = CuArray(c)
        cudacall(vadd, 10, 1, (DevicePtr{Cfloat},DevicePtr{Cfloat},DevicePtr{Cfloat}), ad, bd, cd)
        c = Array(cd)
        @test c ≈ a+b
    end

    # Subtraction
    let
        c = zeros(Float32, 10)
        cd = CuArray(c)
        cudacall(vsub, 10, 1, (DevicePtr{Cfloat},DevicePtr{Cfloat},DevicePtr{Cfloat}), ad, bd, cd)
        c = Array(cd)
        @test c ≈ a-b
    end

    # Multiplication
    let
        c = zeros(Float32, 10)
        cd = CuArray(c)
        cudacall(vmul, 10, 1, (DevicePtr{Cfloat},DevicePtr{Cfloat},DevicePtr{Cfloat}), ad, bd, cd)
        c = Array(cd)
        @test c ≈ a.*b
    end

    # Division
    let
        c = zeros(Float32, 10)
        cd = CuArray(c)
        cudacall(vdiv, 10, 1, (DevicePtr{Cfloat},DevicePtr{Cfloat},DevicePtr{Cfloat}), ad, bd, cd)
        c = Array(cd)
        @test c ≈ a./b
    end
end

end
