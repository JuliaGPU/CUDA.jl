@testset "execution" begin

let
    # test outer CuDim3 constructors
    @test CUDA.CuDim3((Cuint(4),Cuint(3),Cuint(2))) == CUDA.CuDim3(Cuint(4),Cuint(3),Cuint(2))
    @test CUDA.CuDim3((Cuint(3),Cuint(2)))          == CUDA.CuDim3(Cuint(3),Cuint(2),Cuint(1))
    @test CUDA.CuDim3((Cuint(2),))                  == CUDA.CuDim3(Cuint(2),Cuint(1),Cuint(1))
    @test CUDA.CuDim3(Cuint(2))                     == CUDA.CuDim3(Cuint(2),Cuint(1),Cuint(1))

    # outer constructor should type convert
    @test CUDA.CuDim3(2)       == CUDA.CuDim3(Cuint(2),Cuint(1),Cuint(1))
    @test_throws InexactError CUDA.CuDim3(typemax(Int64))

    # CuDim type alias should accept conveniently-typed dimensions
    @test isa(2,        CUDA.CuDim)
    @test isa((2,),     CUDA.CuDim)
    @test isa((2,2),    CUDA.CuDim)
    @test isa((2,2,2),  CUDA.CuDim)
    @test isa(Cuint(2), CUDA.CuDim)
end

@testset "device" begin

let
    md = CuModuleFile(joinpath(@__DIR__, "ptx/dummy.ptx"))
    dummy = CuFunction(md, "dummy")

    # different cudacall syntaxes
    cudacall(dummy, Tuple{})
    cudacall(dummy, Tuple{}; threads=1)
    cudacall(dummy, Tuple{}; threads=1, blocks=1)
    cudacall(dummy, Tuple{}; threads=1, blocks=1, shmem=0)
    cudacall(dummy, Tuple{}; threads=1, blocks=1, shmem=0, stream=CuDefaultStream())
    cudacall(dummy, Tuple{}; threads=1, blocks=1, shmem=0, stream=CuDefaultStream(), cooperative=false)
    cudacall(dummy, ())
    cudacall(dummy, (); threads=1, blocks=1, shmem=0, stream=CuDefaultStream(), cooperative=false)

    # different launch syntaxes
    CUDA.launch(dummy)
    CUDA.launch(dummy; threads=1)
    CUDA.launch(dummy; threads=1, blocks=1)
    CUDA.launch(dummy; threads=1, blocks=1, shmem=0)
    CUDA.launch(dummy; threads=1, blocks=1, shmem=0, stream=CuDefaultStream())
    CUDA.launch(dummy; threads=1, blocks=1, shmem=0, stream=CuDefaultStream(), cooperative=false)
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
        c_d = CuArray(c)
        cudacall(vadd,
                 (CuPtr{Cfloat},CuPtr{Cfloat},CuPtr{Cfloat}), ad, bd, c_d;
                 threads=10)
        c = Array(c_d)
        @test c ≈ a+b
    end

    # Subtraction
    let
        c = zeros(Float32, 10)
        c_d = CuArray(c)
        cudacall(vsub,
                 (CuPtr{Cfloat},CuPtr{Cfloat},CuPtr{Cfloat}), ad, bd, c_d;
                 threads=10)
        c = Array(c_d)
        @test c ≈ a-b
    end

    # Multiplication
    let
        c = zeros(Float32, 10)
        c_d = CuArray(c)
        cudacall(vmul,
                 (CuPtr{Cfloat},CuPtr{Cfloat},CuPtr{Cfloat}), ad, bd, c_d;
                 threads=10)
        c = Array(c_d)
        @test c ≈ a.*b
    end

    # Division
    let
        c = zeros(Float32, 10)
        c_d = CuArray(c)
        cudacall(vdiv,
                 (CuPtr{Cfloat},CuPtr{Cfloat},CuPtr{Cfloat}), ad, bd, c_d;
                 threads=10)
        c = Array(c_d)
        @test c ≈ a./b
    end
end

end

@testset "host" begin
    c = Condition()
    CUDA.launch() do
        notify(c)
    end
    wait(c)
end

@testset "attributes" begin

md = CuModuleFile(joinpath(@__DIR__, "ptx/dummy.ptx"))
dummy = CuFunction(md, "dummy")

val = attributes(dummy)[CUDA.FUNC_ATTRIBUTE_SHARED_SIZE_BYTES]

if CUDA.version() >= v"9.0"
    attributes(dummy)[CUDA.FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES] = val
end

end

end
