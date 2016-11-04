using CUDAdrv

using Compat

## pointer

# construction
@test_throws InexactError DevicePtr(C_NULL)

# conversion
Base.unsafe_convert(Ptr{Void}, CU_NULL)
@test_throws InexactError convert(Ptr{Void}, CU_NULL)
@test_throws InexactError convert(DevicePtr{Void}, C_NULL)

let
    @test eltype(DevicePtr{Void}) == Void
    @test eltype(CU_NULL) == Void
    @test isnull(CU_NULL)

    @test_throws InexactError convert(Ptr{Void}, CU_NULL)
    @test_throws InexactError convert(DevicePtr{Void}, C_NULL)
end


## errors

let
    ex = CuError(0)
    @test CUDAdrv.name(ex) == :SUCCESS
    @test CUDAdrv.description(ex) == "Success"
    
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


## base

CUDAdrv.@apicall(:cuDriverGetVersion, (Ptr{Cint},), Ref{Cint}())

@test_throws ErrorException CUDAdrv.@apicall(:cuNonexisting, ())

@test_throws ErrorException CUDAdrv.@apicall(:cuDummyAvailable, ())
@test_throws CUDAdrv.CuVersionError CUDAdrv.@apicall(:cuDummyUnavailable, ())

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

dev = CuDevice(0)
ctx = CuContext(dev)

@test ctx == CuCurrentContext()


## version

CUDAdrv.version()


## devices

name(dev)
totalmem(dev)
attribute(dev, CUDAdrv.MAX_THREADS_PER_BLOCK)
@test warpsize(dev) == Int32(32)
capability(dev)
list_devices()


## context

@test device(ctx) == dev
synchronize(ctx)
synchronize()

let
    ctx2 = CuContext(dev)       # implicitly pushes
    @test CuCurrentContext() == ctx2
    @test_throws ArgumentError device(ctx)

    push(ctx)
    @test CuCurrentContext() == ctx

    pop()
    @test CuCurrentContext() == ctx2

    pop()
    @test CuCurrentContext() == ctx
end


## module

let
    md = CuModuleFile(joinpath(dirname(@__FILE__), "ptx/vadd.ptx"))

    vadd = CuFunction(md, "vadd")

    unload(md)
end

CuModuleFile(joinpath(dirname(@__FILE__), "ptx/vadd.ptx")) do md
    vadd = CuFunction(md, "vadd")
end

let
    f = open(joinpath(dirname(@__FILE__), "ptx/vadd.ptx"))
    ptx = readstring(f)
    close(f)

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
    md = CuModuleFile(joinpath(dirname(@__FILE__), "ptx/global.ptx"))

    var = CuGlobal{Int32}(md, "foobar")
    @test eltype(var) == Int32
    @test eltype(typeof(var)) == Int32

    @test_throws ArgumentError CuGlobal{Int64}(md, "foobar")

    set(var, Int32(42))
    @test get(var) == Int32(42)

    unload(md)
end

let
    link = CuLink()

    # regular string
    open(joinpath(dirname(@__FILE__), "ptx/empty.ptx")) do f
        addData(link, "vadd_parent", readstring(f), CUDAdrv.PTX)
    end

    # string as vector of bytes
    open(joinpath(dirname(@__FILE__), "ptx/empty.ptx")) do f
        addData(link, "vadd_parent", convert(Vector{UInt8}, readstring(f)), CUDAdrv.PTX)
    end

    # string containing \0
    @test_throws ArgumentError addData(link, "vadd_parent", "\0", CUDAdrv.PTX)
    @test_throws CuError addData(link, "vadd_parent", "\0", CUDAdrv.OBJECT)

    # vector of bytes containing \0
    @test_throws ArgumentError addData(link, "vadd_parent", convert(Vector{UInt8}, "\0"), CUDAdrv.PTX)
    @test_throws CuError addData(link, "vadd_parent", convert(Vector{UInt8}, "\0"), CUDAdrv.OBJECT)

    destroy(link)
end

let
    link = CuLink()
    addFile(link, joinpath(dirname(@__FILE__), "ptx/vadd_child.ptx"), CUDAdrv.PTX)
    open(joinpath(dirname(@__FILE__), "ptx/vadd_parent.ptx")) do f
        addData(link, "vadd_parent", readstring(f), CUDAdrv.PTX)
    end
    obj = complete(link)

    md = CuModule(obj)
    destroy(link)

    vadd = CuFunction(md, "vadd")
    unload(md)
end


## memory

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
    dev_array = CuArray{Int32}(10)
    cumemset(dev_array.devptr, UInt32(0), 10)
    host_array = Array(dev_array)

    @test all(x -> x==0, host_array)
end


## stream

let
    s = CuStream()
    synchronize(s)
    destroy(s)

    synchronize(CuDefaultStream())
end


## execution

let
    @test CUDAdrv.CuDim3((4,3,2)) == CUDAdrv.CuDim3(4,3,2)
    @test CUDAdrv.CuDim3((3,2))   == CUDAdrv.CuDim3(3,2,1)
    @test CUDAdrv.CuDim3((2,))    == CUDAdrv.CuDim3(2,1,1)
    @test CUDAdrv.CuDim3(2)       == CUDAdrv.CuDim3(2,1,1)
end

let
    md = CuModuleFile(joinpath(dirname(@__FILE__), "ptx/vectorops.ptx"))
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

    unload(md)
end


## events

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


## array

let
    # inner constructors
    a = CuArray{Int,1}((2,))
    devptr = a.devptr
    CuArray{Int,1}((2,), devptr)

    # outer constructors
    CuArray{Int}(2)
    CuArray{Int}((1,2))

    # similar
    let a = CuArray{Int}((1,2))
        similar(a)
        similar(a, Float32)
        similar(a, 2)
        similar(a, (2,1))
        similar(a, Float32, 2)
        similar(a, Float32, (2,1))
    end

    # conversions
    @test Base.unsafe_convert(DevicePtr{Int}, CuArray{Int,1}((1,), devptr)) == devptr
    @test pointer(CuArray{Int,1}((1,), devptr)) == devptr

    # negative test cases
    a = rand(Float32, 10)
    ad = CuArray{Float32}(5)
    @test_throws ArgumentError copy!(ad, a)
    @test_throws ArgumentError copy!(a, ad)

    # utility
    @test ndims(ad) == 1
    @test size(ad, 1) == 5
    @test size(ad, 2) == 1
    @test eltype(ad) == Float32
    @test eltype(typeof(ad)) == Float32
end

let
    # ghost type
    @test_throws ArgumentError CuArray([x->x*x for i=1:10])

    # non-isbits elements
    @test_throws ArgumentError CuArray(["foobar" for i=1:10])
    @test_throws ArgumentError CuArray{Function}(10)
    @test_throws ArgumentError CuArray{Function}((10, 10))
end


## profile

@cuprofile begin end


## gc

ctx = nothing
for i in 1:5
    gc()
end

@test length(CUDAdrv.context_consumers) == 0
@test length(CUDAdrv.context_instances) == 0
