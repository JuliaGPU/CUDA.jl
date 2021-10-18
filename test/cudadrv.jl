@testset "context" begin

ctx = current_context()
dev = current_device()

synchronize(ctx)

let ctx2 = CuContext(dev)
    @test ctx2 == current_context()    # ctor implicitly pushes
    activate(ctx)
    @test ctx == current_context()

    @test device(ctx2) == dev

    CUDA.unsafe_destroy!(ctx2)
end

let global_ctx2 = nothing
    CuContext(dev) do ctx2
        @test ctx2 == current_context()
        @test ctx != ctx2
        global_ctx2 = ctx2
    end
    @test !CUDA.isvalid(global_ctx2)
    @test ctx == current_context()

    @test device(ctx) == dev
    @test current_device() == dev
    device_synchronize()
end

end

# FIXME: these trample over our globally-managed context

# @testset "primary context" begin

# pctx = CuPrimaryContext(device())

# @test !isactive(pctx)
# unsafe_reset!(pctx)
# @test !isactive(pctx)

# @test flags(pctx) == 0
# setflags!(pctx, CUDA.CTX_SCHED_BLOCKING_SYNC)
# @test flags(pctx) == CUDA.CTX_SCHED_BLOCKING_SYNC

# let global_ctx = nothing
#     CuContext(pctx) do ctx
#         @test CUDA.isvalid(ctx)
#         @test isactive(pctx)
#         global_ctx = ctx
#     end
#     @test !isactive(pctx)
#     @test !CUDA.isvalid(global_ctx)
# end

# CuContext(pctx) do ctx
#     @test CUDA.isvalid(ctx)
#     @test isactive(pctx)

#     unsafe_reset!(pctx)

#     @test !isactive(pctx)
#     @test !CUDA.isvalid(ctx)
# end

# let
#     @test !isactive(pctx)

#     ctx1 = CuContext(pctx)
#     @test isactive(pctx)
#     @test CUDA.isvalid(ctx1)

#     unsafe_reset!(pctx)
#     @test !isactive(pctx)
#     @test !CUDA.isvalid(ctx1)
#     CUDA.valid_contexts

#     ctx2 = CuContext(pctx)
#     @test isactive(pctx)
#     @test !CUDA.isvalid(ctx1)
#     @test CUDA.isvalid(ctx2)

#     unsafe_reset!(pctx)
# end

# end


@testset "cache config" begin

config = cache_config()

cache_config!(CUDA.FUNC_CACHE_PREFER_L1)
@test cache_config() == CUDA.FUNC_CACHE_PREFER_L1

cache_config!(config)

end


@testset "shmem config" begin

config = shmem_config()

shmem_config!(CUDA.SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE)
@test shmem_config() == CUDA.SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE

shmem_config!(config)

end


@testset "limits" begin

lim = limit(CUDA.LIMIT_DEV_RUNTIME_SYNC_DEPTH)

lim += 1
limit!(CUDA.LIMIT_DEV_RUNTIME_SYNC_DEPTH, lim)
@test lim == limit(CUDA.LIMIT_DEV_RUNTIME_SYNC_DEPTH)

limit!(CUDA.LIMIT_DEV_RUNTIME_SYNC_DEPTH, lim)

end


############################################################################################

@testset "devices" begin

dev = device()

@test name(dev) isa String
@test uuid(dev) isa Base.UUID
@test parent_uuid(dev) isa Base.UUID
totalmem(dev)
attribute(dev, CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
@test warpsize(dev) == 32
capability(dev)
@grab_output show(stdout, "text/plain", dev)

@test eval(Meta.parse(repr(dev))) == dev

@test eltype(devices()) == CuDevice
@grab_output show(stdout, "text/plain", CUDA.DEVICE_CPU)
@grab_output show(stdout, "text/plain", CUDA.DEVICE_INVALID)

@test length(devices()) == ndevices()
@grab_output show(stdout, "text/plain", devices())

end

############################################################################################

@testset "errors" begin

let
    ex = CuError(CUDA.SUCCESS)
    @test CUDA.name(ex) == "SUCCESS"
    @test CUDA.description(ex) == "no error"
    @test eval(Meta.parse(repr(ex))) == ex

    io = IOBuffer()
    showerror(io, ex)
    str = String(take!(io))

    @test occursin("0", str)
    @test occursin("no error", str)
end

let
    ex = CuError(CUDA.SUCCESS, "foobar")

    io = IOBuffer()
    showerror(io, ex)
    str = String(take!(io))

    @test occursin("foobar", str)
end

end

############################################################################################

@testset "events" begin

let
    start = CuEvent()
    stop = CuEvent()
    @test start != stop

    record(start)
    record(stop)
    synchronize(stop)

    @test elapsed(start, stop) > 0
end

@test (CUDA.@elapsed begin end) > 0

CuEvent(CUDA.EVENT_BLOCKING_SYNC)
CuEvent(CUDA.EVENT_BLOCKING_SYNC | CUDA.EVENT_DISABLE_TIMING)

@testset "stream wait" begin
    event  = CuEvent()
    stream = CuStream()

    CUDA.record(event, stream)

    CUDA.wait(event)
    synchronize()
end

@testset "event query" begin
    event = CuEvent()
    @test CUDA.isdone(event)
end

end

############################################################################################

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
    cudacall(dummy, Tuple{}; threads=1, blocks=1, shmem=0, stream=stream())
    cudacall(dummy, Tuple{}; threads=1, blocks=1, shmem=0, stream=stream(), cooperative=false)
    cudacall(dummy, ())
    cudacall(dummy, (); threads=1, blocks=1, shmem=0, stream=stream(), cooperative=false)

    # different launch syntaxes
    CUDA.launch(dummy)
    CUDA.launch(dummy; threads=1)
    CUDA.launch(dummy; threads=1, blocks=1)
    CUDA.launch(dummy; threads=1, blocks=1, shmem=0)
    CUDA.launch(dummy; threads=1, blocks=1, shmem=0, stream=stream())
    CUDA.launch(dummy; threads=1, blocks=1, shmem=0, stream=stream(), cooperative=false)
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

############################################################################################

@testset "graph" begin

let A = CUDA.zeros(Int, 1)
    # ensure compilation
    A .+= 1
    @test Array(A) == [1]

    graph = capture() do
        @test is_capturing()
        A .+= 1
    end
    @test Array(A) == [1]

    exec = instantiate(graph)
    CUDA.launch(exec)
    @test Array(A) == [2]

    graph′ = capture() do
        A .+= 2
    end

    update(exec, graph′)
    CUDA.launch(exec)
    @test Array(A) == [4]
end

let A = CUDA.zeros(Int, 1)
    function iteration(A, val)
        # custom kernel to force compilation on the first iteration
        function kernel(a, val)
            a[] += val
            return
        end
        @cuda kernel(A, val)
        return
    end

    for i in 1:2
        @captured iteration(A, i)
    end
    @test Array(A) == [3]
end

end

############################################################################################

@testset "memory" begin

let
    a,b = Mem.info()
    # NOTE: actually testing this is pretty fragile on CI
    #=@test a == =# CUDA.available_memory()
    #=@test b == =# CUDA.total_memory()
end

# dummy data
T = UInt32
N = 5
data = rand(T, N)
nb = sizeof(data)

# buffers are untyped, so we use a convenience function to get a typed pointer
# we prefer to return a device pointer (for managed buffers) to maximize CUDA coverage
typed_pointer(buf::Union{Mem.Device, Mem.Unified}, T) = convert(CuPtr{T}, buf)
typed_pointer(buf::Mem.Host, T)                       = convert(Ptr{T},   buf)

# allocations and copies
for srcTy in [Mem.Device, Mem.Host, Mem.Unified],
    dstTy in [Mem.Device, Mem.Host, Mem.Unified]

    dummy = Mem.alloc(srcTy, 0)
    Mem.free(dummy)

    src = Mem.alloc(srcTy, nb)
    unsafe_copyto!(typed_pointer(src, T), pointer(data), N)

    dst = Mem.alloc(dstTy, nb)
    unsafe_copyto!(typed_pointer(dst, T), typed_pointer(src, T), N)

    ref = Array{T}(undef, N)
    unsafe_copyto!(pointer(ref), typed_pointer(dst, T), N)

    @test data == ref

    if isa(src, Mem.Device) || isa(src, Mem.Unified)
        Mem.set!(typed_pointer(src, T), zero(T), N)
    end

    # test the memory-type attribute
    if isa(src, Mem.Device)
        @test CUDA.memory_type(typed_pointer(src, T)) == CUDA.MEMORYTYPE_DEVICE
    elseif isa(src, Mem.Host)
        @test CUDA.memory_type(convert(Ptr{T}, src)) == CUDA.MEMORYTYPE_HOST
    elseif isa(src, Mem.Unified)
        # unified memory can reside in either place
        # FIXME: does this depend on the current migration, or on the configuration?
        @test CUDA.memory_type(convert(CuPtr{T}, src)) == CUDA.MEMORYTYPE_HOST ||
              CUDA.memory_type(convert(CuPtr{T}, src)) == CUDA.MEMORYTYPE_DEVICE ||
        @test CUDA.memory_type(convert(CuPtr{T}, src)) == CUDA.memory_type(convert(Ptr{T}, src))
    end

    # test device with context in which pointer was allocated.
    @test device(typed_pointer(src, T)) == device()
    if !CUDA.has_stream_ordered(device())
        # NVIDIA bug #3319609
        @test CuContext(typed_pointer(src, T)) == context()
    end

    # test the is-managed attribute
    if isa(src, Mem.Unified)
        @test CUDA.is_managed(convert(Ptr{T}, src))
        @test CUDA.is_managed(convert(CuPtr{T}, src))
    else
        @test !CUDA.is_managed(typed_pointer(src, T))
    end
    # Test conversion to Ptr throwing an error
    if isa(src, Mem.Device)
        @test_throws ArgumentError convert(Ptr, src)
    end

    @grab_output show(stdout, src)
    @grab_output show(stdout, dst)
    Mem.free(src)
    Mem.free(dst)
end

# pointer attributes
let
    src = Mem.alloc(Mem.Device, nb)

    attribute!(typed_pointer(src, T), CUDA.POINTER_ATTRIBUTE_SYNC_MEMOPS, 0)

    Mem.free(src)
end

# asynchronous operations
let
    src = Mem.alloc(Mem.Device, nb)

    unsafe_copyto!(typed_pointer(src, T), pointer(data), N; async=true)

    Mem.set!(typed_pointer(src, T), zero(T), N; stream=stream())

    Mem.free(src)
end

# pinned memory
let
    # create a pinned and mapped buffer
    src = Mem.alloc(Mem.Host, nb, Mem.HOSTALLOC_DEVICEMAP)

    # get the CPU address and copy some data to the buffer
    cpu_ptr = convert(Ptr{T}, src)
    @test CUDA.memory_type(cpu_ptr) == CUDA.MEMORYTYPE_HOST
    unsafe_copyto!(cpu_ptr, pointer(data), N)

    # get the GPU address and copy back the data
    gpu_ptr = convert(CuPtr{T}, src)
    ref = Array{T}(undef, N)
    unsafe_copyto!(pointer(ref), gpu_ptr, N)
    @test ref == data

    Mem.free(src)
    # NOTE: don't free dst, it's just a mapped pointer
end

# pinned memory with existing memory
if attribute(device(), CUDA.DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED) != 0
    # register a pinned and mapped buffer
    src = Mem.register(Mem.Host, pointer(data), nb, Mem.HOSTREGISTER_DEVICEMAP)

    # get the GPU address and copy back the data
    gpu_ptr = convert(CuPtr{T}, src)
    ref = Array{T}(undef, N)
    unsafe_copyto!(pointer(ref), gpu_ptr, N)
    @test ref == data

    Mem.unregister(src)
end

# unified memory
let
    src = Mem.alloc(Mem.Unified, nb)

    @test_throws BoundsError Mem.prefetch(src, 2*nb; device=CUDA.DEVICE_CPU)
    # FIXME: prefetch doesn't work on some CI devices, unsure why.
    @test_skip Mem.prefetch(src, nb; device=CUDA.DEVICE_CPU)
    Mem.advise(src, Mem.ADVISE_SET_READ_MOSTLY)

    # get the CPU address and copy some data
    cpu_ptr = convert(Ptr{T}, src)
    unsafe_copyto!(cpu_ptr, pointer(data), N)

    # get the GPU address and copy back data
    gpu_ptr = convert(CuPtr{T}, src)
    ref = Array{T}(undef, N)
    unsafe_copyto!(pointer(ref), gpu_ptr, N)
    @test ref == data

    Mem.free(src)
end

# 3d memcpy
let
    # TODO: use cuMemAllocPitch (and put pitch in buffer?) to actually get benefit from this

    data = collect(reshape(1:27, 3, 3, 3))

    dst = Mem.alloc(Mem.Device, sizeof(data))
    Mem.unsafe_copy3d!(typed_pointer(dst, Int), Mem.Device, pointer(data), Mem.Host, length(data))

    check = zeros(Int, size(data))
    Mem.unsafe_copy3d!(pointer(check), Mem.Host, typed_pointer(dst, Int), Mem.Device, length(data))

    @test check == data

    Mem.free(dst)
end
let
    # copying an x-z plane of a 3-D array

    T = Int
    nx, ny, nz = 4, 4, 4
    data = collect(reshape(1:(nx*nz), nx, nz))
    dst = Mem.alloc(Mem.Device, nx*ny*nz*sizeof(data))

    # host to device
    Mem.unsafe_copy3d!(typed_pointer(dst, T), Mem.Device, pointer(data), Mem.Host,
                       nx, 1, nz;
                       dstPos=(1,2,1),
                       srcPitch=nx*sizeof(T), srcHeight=1,
                       dstPitch=nx*sizeof(T), dstHeight=ny)

    # copy back
    check = zeros(T, size(data))
    Mem.unsafe_copy3d!(pointer(check), Mem.Host, typed_pointer(dst, T), Mem.Device,
                       nx, 1, nz;
                       srcPos=(1,2,1),
                       srcPitch=nx*sizeof(T), srcHeight=ny,
                       dstPitch=nx*sizeof(T), dstHeight=1)

    @test check == data

    # copy back into a 3-D array
    check2 = zeros(T, nx, ny, nz)
    Mem.unsafe_copy3d!(pointer(check2), Mem.Host, typed_pointer(dst, T), Mem.Device,
                       nx, 1, nz;
                       srcPos=(1,2,1),
                       dstPos=(1,2,1),
                       srcPitch=nx*sizeof(T), srcHeight=ny,
                       dstPitch=nx*sizeof(T), dstHeight=ny)
    @test check2[:,2,:] == data
end
let
    # copying an y-z plane of a 3-D array

    T = Int
    nx, ny, nz = 4, 4, 4
    data = collect(reshape(1:(ny*nz), ny, nz))
    dst = Mem.alloc(Mem.Device, nx*ny*nz*sizeof(data))

    # host to device
    Mem.unsafe_copy3d!(typed_pointer(dst, T), Mem.Device, pointer(data), Mem.Host,
                       1, ny, nz;
                       dstPos=(2,1,1),
                       srcPitch=1*sizeof(T), srcHeight=ny,
                       dstPitch=nx*sizeof(T), dstHeight=ny)

    # copy back
    check = zeros(T, size(data))
    Mem.unsafe_copy3d!(pointer(check), Mem.Host, typed_pointer(dst, T), Mem.Device,
                       1, ny, nz;
                       srcPos=(2,1,1),
                       srcPitch=nx*sizeof(T), srcHeight=ny,
                       dstPitch=1*sizeof(T), dstHeight=ny)

    @test check == data

    # copy back into a 3-D array
    check2 = zeros(T, nx, ny, nz)
    Mem.unsafe_copy3d!(pointer(check2), Mem.Host, typed_pointer(dst, T), Mem.Device,
                       1, ny, nz;
                       srcPos=(2,1,1),
                       dstPos=(2,1,1),
                       srcPitch=nx*sizeof(T), srcHeight=ny,
                       dstPitch=nx*sizeof(T), dstHeight=ny)
    @test check2[2,:,:] == data
end
let
    # JuliaGPU/CUDA.jl#863: wrong offset calculation
    nx, ny, nz = 1, 2, 1

    A = zeros(Int, nx, nz)
    B = CuArray(reshape([1:(nx*ny*nz)...], (nx, ny, nz)))
    Mem.unsafe_copy3d!(
        pointer(A), Mem.Host, pointer(B), Mem.Device,
        nx, 1, nz;
        srcPos=(1,2,1),
        srcPitch=nx*sizeof(A[1]), srcHeight=ny,
        dstPitch=nx*sizeof(A[1]), dstHeight=1
    )

    @test A == Array(B)[:,2,:]
end

# pinned memory with existing memory
if attribute(device(), CUDA.DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED) != 0
    hA = rand(UInt8, 512)
    @test !CUDA.is_pinned(pointer(hA))
    Mem.pin(hA)
    @test CUDA.is_pinned(pointer(hA))

    # make sure we can double-pin
    Mem.pin(hA)

    # memory copies on pinned memory behave differently, so test that code path
    dA = CUDA.rand(UInt8, 512)
    copyto!(dA, hA)
    copyto!(hA, dA)
end

end

############################################################################################

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

@test_throws_cuerror CUDA.ERROR_INVALID_IMAGE CuModule("foobar")

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
        @test_throws_cuerror CUDA.ERROR_UNKNOWN add_data!(link, "vadd_parent", UInt8[0])
    end
end

@testset "error log" begin
    @test_throws_message contains("ptxas fatal") CuError CuModule(".version 3.1")

    link = CuLink()
    @test_throws_message contains("ptxas fatal") CuError add_data!(link, "dummy", ".version 3.1")
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

############################################################################################

@testset "occupancy" begin

let md = CuModuleFile(joinpath(@__DIR__, "ptx/dummy.ptx"))
    dummy = CuFunction(md, "dummy")

    active_blocks(dummy, 1)
    active_blocks(dummy, 1; shmem=64)

    occupancy(dummy, 1)
    occupancy(dummy, 1; shmem=64)

    launch_configuration(dummy)
    launch_configuration(dummy; shmem=64)
    launch_configuration(dummy; shmem=64, max_threads=64)

    let cb_calls = 0
        launch_configuration(dummy; shmem=threads->(cb_calls += 1; 0))
        @test cb_calls > 0
    end
end

end

############################################################################################

@testset "pool" begin

dev = device()
if CUDA.version() >= v"11.2" && attribute(dev, CUDA.DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED) == 1

pool = memory_pool(dev)

pool2 = CuMemoryPool(dev)
@test pool2 != pool
memory_pool!(dev, pool2)
@test pool2 == memory_pool(dev)
@test pool2 != default_memory_pool(dev)

memory_pool!(dev, pool)
@test pool == memory_pool(dev)

@test attribute(UInt64, pool2, CUDA.MEMPOOL_ATTR_RELEASE_THRESHOLD) == 0
attribute!(pool2, CUDA.MEMPOOL_ATTR_RELEASE_THRESHOLD, UInt64(2^30))
@test attribute(UInt64, pool2, CUDA.MEMPOOL_ATTR_RELEASE_THRESHOLD) == 2^30

end

end

############################################################################################

@testset "profile" begin

@test_logs (:warn, r"only informs an external profiler to start") CUDA.Profile.start()
CUDA.Profile.stop()

@test_logs (:warn, r"only informs an external profiler to start") CUDA.@profile begin end

end

############################################################################################

@testset "stream" begin

s = CuStream()
synchronize(s)
@test CUDA.isdone(s)

let s2 = CuStream()
    @test s != s2
    @test !(s == s2)
end

let s3 = CuStream(; flags=CUDA.STREAM_NON_BLOCKING)
    @test s != s3
    @test !(s == s3)
end

prio = priority_range()
let s = CuStream(; priority=first(prio))
    @test priority(s) == first(prio)
end
let s = CuStream(; priority=last(prio))
    @test priority(s) == last(prio)
end

synchronize()
synchronize(; blocking=false)
synchronize(stream())
synchronize(stream(); blocking=false)

@grab_output show(stdout, stream())

end

############################################################################################

@testset "version" begin

@test isa(CUDA.version(), VersionNumber)

@test isa(CUDA.release(), VersionNumber)
@test CUDA.release().patch == 0

end
