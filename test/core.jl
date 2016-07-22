using CUDAdrv

using Compat


## Base

CUDAdrv.@apicall(:cuDriverGetVersion, (Ptr{Cint},), Ref{Cint}())

@test_throws ErrorException CUDAdrv.@apicall(:nonExisting, ())
CUDAdrv.trace(prefix=" ")

@test_throws ErrorException eval(
    quote
        foo = :bar
        CUDAdrv.@apicall(foo, ())
    end
)

typealias CuDevice_t Cint
try
    CUDAdrv.@apicall(:cuDeviceGet, (Ptr{CuDevice_t}, Cint), Ref{CuDevice_t}(), devcount())
catch e
    e == CUDAdrv.ERROR_INVALID_DEVICE || rethrow(e)
end

CUDAdrv.vendor()
CUDAdrv.version()

dev = CuDevice(0)
ctx = CuContext(dev)


## pointer

DevicePtr{Void}()
@test_throws InexactError DevicePtr{Void}(C_NULL)
DevicePtr{Void}(C_NULL, true)

let
    nullptr = DevicePtr{Void}()

    @test eltype(DevicePtr{Void}) == Void
    @test eltype(nullptr) == Void
    @test isnull(nullptr)

    @test_throws InexactError convert(Ptr{Void}, nullptr)
    @test_throws InexactError convert(DevicePtr{Void}, C_NULL)
end


## CuContext

@test device(ctx) == dev
synchronize(ctx)

let
    ctx2 = CuContext(dev)       # implicitly pushes
    @test current_context() == ctx2
    @test_throws ArgumentError device(ctx)

    push(ctx)
    @test current_context() == ctx

    pop()
    @test current_context() == ctx2

    pop()
    @test current_context() == ctx

    destroy(ctx2)
end


## CuDevice

name(dev)
totalmem(dev)
attribute(dev, CUDAdrv.MAX_THREADS_PER_BLOCK)
capability(dev)
list_devices()


## CuError

let
    ex = CuError(0)
    @test name(ex) == :SUCCESS
    @test description(ex) == "Success"
    
    io = IOBuffer()
    showerror(io, ex)
    str = takebuf_string(io)

    @test contains(str, "0")
    @test contains(str, "Success")
end

let
    ex = CuError(0, "foobar")
    
    io = IOBuffer()
    showerror(io, ex)
    str = takebuf_string(io)

    @test contains(str, "foobar")
end


## CuEvent

let
    start = CuEvent()
    stop = CuEvent()
    record(start)
    record(stop)
    synchronize(stop)
    @test elapsed(start, stop) > 0
    destroy(start)
    destroy(stop)
end


## CuArray

let
    # Inner constructors
    CuArray{Int,1}((1,))
    CuArray{Int,1}((1,), DevicePtr{Int}(C_NULL, true))

    # Outer constructors
    CuArray{Int}(1)
    CuArray{Int}((1,2))
    CuArray(Int, 1)
    CuArray(Int, (1,2))

    # Negative test cases
    a = rand(Float32, 10)
    ad = CuArray(Float32, 5)
    @test_throws ArgumentError copy!(ad, a)
    @test_throws ArgumentError copy!(a, ad)

    # Utility
    @test ndims(ad) == 1
    @test size(ad, 1) == 5
    @test size(ad, 2) == 1
    @test eltype(ad) == Float32
    @test eltype(typeof(ad)) == Float32

    free(ad)
end

let
    # ghost type
    @test_throws ArgumentError CuArray([x->x*x for i=1:10])

    # non-isbits elements
    @test_throws ArgumentError CuArray(["foobar" for i=1:10])
    @test_throws ArgumentError CuArray(Function, 10)
    @test_throws ArgumentError CuArray(Function, (10, 10))
end


## CuModule

let
    md = CuModuleFile(joinpath(Base.source_dir(), "vectorops.ptx"))

    vadd = CuFunction(md, "vadd")

    unload(md)
end

CuModuleFile(joinpath(Base.source_dir(), "vectorops.ptx")) do md
    vadd = CuFunction(md, "vadd")
end

let
    f = open(joinpath(Base.source_dir(), "vectorops.ptx"))
    ptx = readstring(f)

    md = CuModule(ptx)
    vadd = CuFunction(md, "vadd")
    unload(md)
end

try
    CuModule("foobar")
catch ex
    ex == CUDAdrv.ERROR_INVALID_IMAGE  || rethrow(ex)
end

let
    md = CuModuleFile(joinpath(Base.source_dir(), "global.ptx"))

    var = CuGlobal{Int32}(md, "foobar")
    @test_throws ArgumentError CuGlobal{Int64}(md, "foobar")

    @test eltype(var) == Int32
    set(var, Int32(42))
    @test get(var) == Int32(42)

    unload(md)
end


## CuProfile

@cuprofile begin end


## CuStream

let
    s = CuStream()
    synchronize(s)
    destroy(s)

    synchronize(default_stream())
end


## memory handling

let
    ptr = cualloc(Int, 1)
    free(ptr)

    @test_throws ArgumentError cualloc(Function, 1)   # abstract
    @test_throws ArgumentError cualloc(Array{Int}, 1) # non-leaftype
    @test_throws ArgumentError cualloc(Int, 0)
end

let
    type MutablePtrFree
        foo::Int
        bar::Int
    end
    ptr = cualloc(MutablePtrFree, 1)
    copy!(ptr, MutablePtrFree(0,0))
    free(ptr)
end

let
    type MutableNonPtrFree
        foo::Int
        bar::String
    end
    ptr = cualloc(MutableNonPtrFree, 1)
    @test_throws ArgumentError copy!(ptr, MutableNonPtrFree(0,""))
    free(ptr)
end


let
    dev_array = CuArray(Int32, 10)
    cumemset(dev_array.ptr, UInt32(0), 10)
    host_array = Array(dev_array)

    @test all(x -> x==0, host_array)

    free(dev_array)
end


## PTX loading & execution

let
    @test CUDAdrv.CuDim3((3,2,1)) == CUDAdrv.CuDim3(3,2,1)
    @test CUDAdrv.CuDim3((3,2))   == CUDAdrv.CuDim3(3,2,1)
    @test CUDAdrv.CuDim3(3)       == CUDAdrv.CuDim3(3,1,1)
end

let
    md = CuModuleFile(joinpath(Base.source_dir(), "vectorops.ptx"))
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
        free(cd)
    end

    # Subtraction
    let
        c = zeros(Float32, 10)
        cd = CuArray(c)
        cudacall(vsub, 10, 1, (DevicePtr{Cfloat},DevicePtr{Cfloat},DevicePtr{Cfloat}), ad, bd, cd)
        c = Array(cd)
        @test c ≈ a-b
        free(cd)
    end

    # Multiplication
    let
        c = zeros(Float32, 10)
        cd = CuArray(c)
        cudacall(vmul, 10, 1, (DevicePtr{Cfloat},DevicePtr{Cfloat},DevicePtr{Cfloat}), ad, bd, cd)
        c = Array(cd)
        @test c ≈ a.*b
        free(cd)
    end

    # Division
    let
        c = zeros(Float32, 10)
        cd = CuArray(c)
        cudacall(vdiv, 10, 1, (DevicePtr{Cfloat},DevicePtr{Cfloat},DevicePtr{Cfloat}), ad, bd, cd)
        c = Array(cd)
        @test c ≈ a./b
        free(cd)
    end

    free(ad)
    free(bd)
    unload(md)
end


destroy(ctx)
