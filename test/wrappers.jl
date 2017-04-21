@testset "API wrappers" begin

@testset "version" begin

@test isa(CUDAdrv.version(), VersionNumber)

end


@testset "devices" begin

name(dev)
totalmem(dev)
attribute(dev, CUDAdrv.MAX_THREADS_PER_BLOCK)
@test warpsize(dev) == Int32(32)
capability(dev)
@grab_output list_devices()

end


@testset "context" begin

@test ctx == CuCurrentContext()
@test ctx === CuCurrentContext()

@test_throws ErrorException deepcopy(ctx)

let ctx2 = CuContext(dev)
    @test ctx2 == CuCurrentContext()    # ctor implicitly pushes
    activate(ctx)
    @test ctx == CuCurrentContext()

    @test_throws ErrorException device(ctx2)

    destroy(ctx2)
end

instances = length(CUDAdrv.context_instances)
CuContext(dev) do ctx2
    @test length(CUDAdrv.context_instances) == instances+1
    @test ctx2 == CuCurrentContext()
    @test ctx != ctx2
end
@test length(CUDAdrv.context_instances) == instances
@test ctx == CuCurrentContext()

@test device(ctx) == dev
synchronize(ctx)
synchronize()

end

@testset "primary context" begin

pctx = CuPrimaryContext(0)

state(pctx)
@test !isactive(pctx)

@test flags(pctx) == CUDAdrv.SCHED_AUTO
setflags!(pctx, CUDAdrv.SCHED_BLOCKING_SYNC)
@test flags(pctx) == CUDAdrv.SCHED_BLOCKING_SYNC

CuContext(pctx) do ctx
    @test isactive(pctx)
end
gc()
@test !isactive(pctx)

end

@testset "module" begin

let
    md = CuModuleFile(joinpath(@__DIR__, "ptx/vadd.ptx"))

    vadd = CuFunction(md, "vadd")
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

let
    md = CuModuleFile(joinpath(@__DIR__, "ptx/global.ptx"))

    var = CuGlobal{Int32}(md, "foobar")
    @test eltype(var) == Int32
    @test eltype(typeof(var)) == Int32

    @test_throws ArgumentError CuGlobal{Int64}(md, "foobar")

    set(var, Int32(42))
    @test get(var) == Int32(42)
end

let
    link = CuLink()

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


@testset "memory" begin

# pointer-based
let
    obj = 42

    ptr = Mem.alloc(sizeof(obj))

    Mem.set(ptr, Cuint(0), sizeof(Int)÷sizeof(Cuint))

    Mem.upload(ptr, Ref(obj), sizeof(obj))

    obj_copy = Ref(0)
    Mem.download(Ref(obj_copy), ptr, sizeof(obj))
    @test obj == obj_copy[]

    ptr2 = Mem.alloc(sizeof(obj))
    Mem.transfer(ptr2, ptr, sizeof(obj))
    obj_copy2 = Ref(0)
    Mem.download(Ref(obj_copy2), ptr2, sizeof(obj))
    @test obj == obj_copy2[]

    Mem.free(ptr2)
    Mem.free(ptr)
end

let
    dev_array = CuArray{Int32}(10)
    Mem.set(dev_array.devptr, UInt32(0), 10)
    host_array = Array(dev_array)

    @test all(x -> x==0, host_array)
end

# object-based
let
    obj = 42

    ptr = Mem.alloc(typeof(obj))

    Mem.upload(ptr, obj)

    obj_copy = Mem.download(ptr)
    @test obj == obj_copy

    ptr2 = Mem.upload(obj)

    obj_copy2 = Mem.download(ptr2)
    @test obj == obj_copy2
end

let
    @test_throws ArgumentError Mem.alloc(Function, 1)   # abstract
    @test_throws ArgumentError Mem.alloc(Array{Int}, 1) # UnionAll
    @test_throws ArgumentError Mem.alloc(Integer, 1)    # abstract
    # TODO: can we test for the third case?
    #       !abstract && leaftype seems to imply UnionAll nowadays...
    @test_throws ArgumentError Mem.alloc(Int, 0)

    # double-free should throw (we rely on it for CuArray finalizer tests)
    x = Mem.alloc(1)
    Mem.free(x)
    @test_throws_cuerror CUDAdrv.ERROR_INVALID_VALUE Mem.free(x)
end

let
    @eval type MutablePtrFree
        foo::Int
        bar::Int
    end
    ptr = Mem.alloc(MutablePtrFree, 1)
    Mem.upload(ptr, MutablePtrFree(0,0))
    Mem.free(ptr)
end

let
    @eval type MutableNonPtrFree
        foo::Int
        bar::String
    end
    ptr = Mem.alloc(MutableNonPtrFree, 1)
    @test_throws ArgumentError Mem.upload(ptr, MutableNonPtrFree(0,""))
    Mem.free(ptr)
end

end


@testset "stream" begin

let
    s = CuStream()
    synchronize(s)
    let s2 = CuStream()
        @test s != s2
    end

    synchronize(CuDefaultStream())
end

end


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


@testset "events" begin

let
    start = CuEvent()
    stop = CuEvent()
    @test start != stop
    record(start)
    record(stop)
    synchronize(stop)
    @test elapsed(start, stop) > 0
    @test (CUDAdrv.@elapsed begin
        end) > 0
    @test (CUDAdrv.@elapsed CuDefaultStream() begin
        end) > 0
end

end


@testset "array" begin

let
    # inner constructors
    let
        arr = CuArray{Int,1}((2,))
        devptr = arr.devptr
        CuArray{Int,1}((2,), devptr)
    end

    # outer constructors
    for I in [Int32,Int64]
        a = I(1)
        b = I(2)

        # partially parameterized
        CuArray{I}(b)
        CuArray{I}((b,))
        CuArray{I}(a,b)
        CuArray{I}((a,b))

        # fully parameterized
        CuArray{I,1}(b)
        CuArray{I,1}((b,))
        @test_throws MethodError CuArray{I,1}(a,b)
        @test_throws MethodError CuArray{I,1}((a,b))
        @test_throws MethodError CuArray{I,2}(b)
        @test_throws MethodError CuArray{I,2}((b,))
        CuArray{I,2}(a,b)
        CuArray{I,2}((a,b))
    end

    # similar
    let a = CuArray{Int}(2)
        similar(a)
        similar(a, Float32)
    end
    let a = CuArray{Int}((1,2))
        similar(a)
        similar(a, Float32)
        similar(a, 2)
        similar(a, (2,1))
        similar(a, Float32, 2)
        similar(a, Float32, (2,1))
    end

    # conversions
    let
        devptr = convert(DevicePtr{Int}, CU_NULL)
        @test Base.unsafe_convert(DevicePtr{Int}, CuArray{Int,1}((1,), devptr)) == devptr
        @test pointer(CuArray{Int,1}((1,), devptr)) == devptr
    end

    # copy: size mismatches
    let
        a = rand(Float32, 10)
        ad = CuArray{Float32}(5)
        bd = CuArray{Float32}(10)

        @test_throws ArgumentError copy!(ad, a)
        @test_throws ArgumentError copy!(a, ad)
        @test_throws ArgumentError copy!(ad, bd)
    end

    # copy to and from device
    let
        cpu = rand(Float32, 10)
        gpu = CuArray{Float32}(10)

        copy!(gpu, cpu)

        cpu_back = Array{Float32}(10)
        copy!(cpu_back, gpu)
        @assert cpu == cpu_back
    end

    # same, but with convenience functions
    let
        cpu = rand(Float32, 10)

        gpu = CuArray(cpu)
        cpu_back = Array(gpu)

        @assert cpu == cpu_back
    end

    # copy on device
    let gpu = CuArray(rand(Float32, 10))
        gpu_copy = copy(gpu)
        @test gpu != gpu_copy
        @test Array(gpu) == Array(gpu_copy)
    end

    # utility
    let gpu = CuArray{Float32}(5)
        @test ndims(gpu) == 1
        @test size(gpu, 1) == 5
        @test size(gpu, 2) == 1
        @test eltype(gpu) == Float32
        @test eltype(typeof(gpu)) == Float32
    end

    # printing
    let gpu = CuArray([42])
        show(DevNull, gpu)
        show(DevNull, "text/plain", gpu)
    end

    # finalizers
    let gpu = CuArray([42])
        finalize(gpu) # triggers early finalization
        finalize(gpu) # shouldn't re-run the finalizer
    end
end

let
    # ghost type
    @test_throws ArgumentError CuArray([x->x*x for i=1:10])

    # non-isbits elements
    @test_throws ArgumentError CuArray(["foobar" for i=1:10])
    @test_throws ArgumentError CuArray{Function}(10)
    @test_throws ArgumentError CuArray{Function}((10, 10))
end

end


@testset "profile" begin

CUDAdrv.Profile.start()
CUDAdrv.Profile.stop()

CUDAdrv.@profile begin end

end

end
