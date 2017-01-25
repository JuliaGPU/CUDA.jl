using CUDAdrv

using Compat


## pointer

# conversion to Ptr
@test_throws InexactError convert(Ptr{Void}, CU_NULL)
Base.unsafe_convert(Ptr{Void}, CU_NULL)

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
    str = String(take!(io))

    @test contains(str, "0")
    @test contains(str, "Success")
end

let
    ex = CuError(0, "foobar")
    
    io = IOBuffer()
    showerror(io, ex)
    str = String(take!(io))

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
ctx = CuContext(dev, CUDAdrv.SCHED_BLOCKING_SYNC)


## version

CUDAdrv.version()


## devices

name(dev)
totalmem(dev)
attribute(dev, CUDAdrv.MAX_THREADS_PER_BLOCK)
@test warpsize(dev) == Int32(32)
capability(dev)
@grab_output list_devices()


## context

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


## module

let
    md = CuModuleFile(joinpath(dirname(@__FILE__), "ptx/vadd.ptx"))

    vadd = CuFunction(md, "vadd")
end

let
    f = open(joinpath(dirname(@__FILE__), "ptx/vadd.ptx"))
    ptx = readstring(f)
    close(f)

    md = CuModule(ptx)
    vadd = CuFunction(md, "vadd")

    md2 = CuModuleFile(joinpath(dirname(@__FILE__), "ptx/vadd.ptx"))
    @test md != md2
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
end

let
    link = CuLink()
    addFile(link, joinpath(dirname(@__FILE__), "ptx/vadd_child.ptx"), CUDAdrv.PTX)
    open(joinpath(dirname(@__FILE__), "ptx/vadd_parent.ptx")) do f
        addData(link, "vadd_parent", readstring(f), CUDAdrv.PTX)
    end

    obj = complete(link)
    md = CuModule(obj)

    vadd = CuFunction(md, "vadd")
end


## memory

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
    @test_throws ArgumentError Mem.alloc(Array{Int}, 1) # non-leaftype
    @test_throws ArgumentError Mem.alloc(Int, 0)
end

let
    type MutablePtrFree
        foo::Int
        bar::Int
    end
    ptr = Mem.alloc(MutablePtrFree, 1)
    Mem.upload(ptr, MutablePtrFree(0,0))
    Mem.free(ptr)
end

let
    type MutableNonPtrFree
        foo::Int
        bar::String
    end
    ptr = Mem.alloc(MutableNonPtrFree, 1)
    @test_throws ArgumentError Mem.upload(ptr, MutableNonPtrFree(0,""))
    Mem.free(ptr)
end


## stream

let
    s = CuStream()
    synchronize(s)
    let s2 = CuStream()
        @test s != s2
    end

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
end


## events

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
    @test Base.unsafe_convert(DevicePtr{Int}, CuArray{Int,1}((1,), devptr)) == devptr
    @test pointer(CuArray{Int,1}((1,), devptr)) == devptr

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

# force garbage collection (this makes finalizers run before STDOUT is destroyed)
destroy(ctx)
for i in 1:5
    gc()
end

# test there's no outstanding contexts or consumers thereof
@test length(CUDAdrv.finalizer_blocks) == 0
@test length(CUDAdrv.context_instances) == 0
