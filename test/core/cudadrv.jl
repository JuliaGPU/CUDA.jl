@testset "context" begin

# perform an API call to ensure everything is initialized
# (or the low-level driver calls below can fail)
synchronize()

ctx = current_context()
@test CUDACore.isvalid(ctx)
@test unique_id(ctx) > 0

dev = current_device()
exclusive = attribute(dev, CUDACore.DEVICE_ATTRIBUTE_COMPUTE_MODE) == CUDACore.CU_COMPUTEMODE_EXCLUSIVE_PROCESS

synchronize(ctx)

@test startswith(sprint(show, MIME"text/plain"(), ctx), "CuContext")
@test CUDACore.api_version(ctx) isa Cuint

if !exclusive
    let ctx2 = CuContext(dev)
        @test ctx2 == current_context()    # ctor implicitly pushes
        activate(ctx)
        @test ctx == current_context()

        @test device(ctx2) == dev

        CUDACore.unsafe_destroy!(ctx2)
    end

    let global_ctx2 = nothing
        CuContext(dev) do ctx2
            @test ctx2 == current_context()
            @test ctx != ctx2
            global_ctx2 = ctx2
        end
        @test !CUDACore.isvalid(global_ctx2)
        @test ctx == current_context()

        @test device(ctx) == dev
        @test current_device() == dev
        device_synchronize()
    end
end

end


@testset "primary context" begin

# we need to start from scratch for these tests
dev = device()
device_reset!()
@test_throws UndefRefError current_context()

pctx = CuPrimaryContext(dev)

@test !isactive(pctx)
unsafe_reset!(pctx)
@test !isactive(pctx)

@test flags(pctx) == 0
setflags!(pctx, CUDACore.CTX_SCHED_BLOCKING_SYNC)
@test flags(pctx) == CUDACore.CTX_SCHED_BLOCKING_SYNC

let global_ctx = nothing
    CuContext(pctx) do ctx
        @test CUDACore.isvalid(ctx)
        @test isactive(pctx)
        global_ctx = ctx
    end
    @test !isactive(pctx) broken=true
    @test !CUDACore.isvalid(global_ctx) broken=true
end

let
    ctx = CuContext(pctx)
    @test CUDACore.isvalid(ctx)
    @test isactive(pctx)

    unsafe_reset!(pctx)

    @test !isactive(pctx)
    @test !CUDACore.isvalid(ctx)
end

let
    @test !isactive(pctx)

    ctx1 = CuContext(pctx)
    @test isactive(pctx)
    @test CUDACore.isvalid(ctx1)

    unsafe_reset!(pctx)
    @test !isactive(pctx)
    @test !CUDACore.isvalid(ctx1)

    ctx2 = CuContext(pctx)
    @test isactive(pctx)
    @test !CUDACore.isvalid(ctx1)
    @test CUDACore.isvalid(ctx2)

    unsafe_reset!(pctx)
end

end


@testset "cache config" begin

config = cache_config()

cache_config!(CUDACore.FUNC_CACHE_PREFER_L1)
@test cache_config() == CUDACore.FUNC_CACHE_PREFER_L1

cache_config!(config)

end


@testset "shmem config" begin

config = shmem_config()

shmem_config!(CUDACore.SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE)
@test shmem_config() == CUDACore.SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE

shmem_config!(config)

end


capability(device()) >= v"9" || # JuliaGPU/CUDA.jl#1846
@testset "limits" begin

lim = limit(CUDACore.LIMIT_DEV_RUNTIME_SYNC_DEPTH)

lim += 1
limit!(CUDACore.LIMIT_DEV_RUNTIME_SYNC_DEPTH, lim)
@test lim == limit(CUDACore.LIMIT_DEV_RUNTIME_SYNC_DEPTH)

limit!(CUDACore.LIMIT_DEV_RUNTIME_SYNC_DEPTH, lim)

end


############################################################################################

@testset "devices" begin

dev = device()

@test name(dev) isa String
@test uuid(dev) isa Base.UUID
@test parent_uuid(dev) isa Base.UUID
totalmem(dev)
attribute(dev, CUDACore.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
@test warpsize(dev) == 32
capability(dev)
@grab_output show(stdout, "text/plain", dev)

@test eval(Meta.parse(repr(dev))) == dev

@test eltype(devices()) == CuDevice
@grab_output show(stdout, "text/plain", CUDACore.DEVICE_CPU)
@grab_output show(stdout, "text/plain", CUDACore.DEVICE_INVALID)

@test length(devices()) == ndevices()
@grab_output show(stdout, "text/plain", devices())

end

############################################################################################

@testset "errors" begin

let
    ex = CuError(CUDACore.SUCCESS)
    @test CUDACore.name(ex) == "SUCCESS"
    @test CUDACore.description(ex) == "no error"
    @test eval(Meta.parse(repr(ex))) == ex

    io = IOBuffer()
    showerror(io, ex)
    str = String(take!(io))

    @test occursin("0", str)
    @test occursin("no error", str)
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

@test (CUDACore.@elapsed identity(nothing)) > 0
@test (CUDACore.@elapsed blocking=true identity(nothing)) > 0

let
    empty_fun() = nothing

    # Check that the benchmarked code occurs as-is in the macro expansion.
    me = @macroexpand1 CUDACore.@elapsed empty_fun()
    @test any(arg -> arg == :(empty_fun()), me.args)
end

CuEvent(CUDACore.EVENT_BLOCKING_SYNC)
CuEvent(CUDACore.EVENT_BLOCKING_SYNC | CUDACore.EVENT_DISABLE_TIMING)

@testset "stream wait" begin
    event  = CuEvent()
    stream = CuStream()

    CUDACore.record(event, stream)

    CUDACore.wait(event)
    synchronize()
end

@testset "event query" begin
    event = CuEvent()
    @test CUDACore.isdone(event)
end

end

############################################################################################

@testset "execution" begin

let
    # test outer CuDim3 constructors
    @test CUDACore.CuDim3((Cuint(4),Cuint(3),Cuint(2))) == CUDACore.CuDim3(Cuint(4),Cuint(3),Cuint(2))
    @test CUDACore.CuDim3((Cuint(3),Cuint(2)))          == CUDACore.CuDim3(Cuint(3),Cuint(2),Cuint(1))
    @test CUDACore.CuDim3((Cuint(2),))                  == CUDACore.CuDim3(Cuint(2),Cuint(1),Cuint(1))
    @test CUDACore.CuDim3(Cuint(2))                     == CUDACore.CuDim3(Cuint(2),Cuint(1),Cuint(1))

    # outer constructor should type convert
    @test CUDACore.CuDim3(2)       == CUDACore.CuDim3(Cuint(2),Cuint(1),Cuint(1))
    @test_throws InexactError CUDACore.CuDim3(typemax(Int64))

    # CuDim type alias should accept conveniently-typed dimensions
    @test isa(2,        CUDACore.CuDim)
    @test isa((2,),     CUDACore.CuDim)
    @test isa((2,2),    CUDACore.CuDim)
    @test isa((2,2,2),  CUDACore.CuDim)
    @test isa(Cuint(2), CUDACore.CuDim)
end

@testset "device" begin

let
    md = CuModuleFile(joinpath(@__DIR__, "ptx/dummy.ptx"))
    dummy = CuFunction(md, "dummy")

    # different cudacall syntaxes
    cudacall(dummy, Tuple{})
    cudacall(dummy, Tuple{}; threads=1)
    cudacall(dummy, Tuple{}; threads=1, blocks=1)
    cudacall(dummy, Tuple{}; threads=1, blocks=1, clustersize=1)
    cudacall(dummy, Tuple{}; threads=1, blocks=1, clustersize=1, shmem=0)
    cudacall(dummy, Tuple{}; threads=1, blocks=1, clustersize=1, shmem=0, stream=stream())
    cudacall(dummy, Tuple{}; threads=1, blocks=1, clustersize=1, shmem=0, stream=stream(), cooperative=false)
    cudacall(dummy, ())
    cudacall(dummy, (); threads=1, blocks=1, clustersize=1, shmem=0, stream=stream(), cooperative=false)

    # different launch syntaxes
    CUDACore.launch(dummy)
    CUDACore.launch(dummy; threads=1)
    CUDACore.launch(dummy; threads=1, blocks=1)
    CUDACore.launch(dummy; threads=1, blocks=1, clustersize=1)
    CUDACore.launch(dummy; threads=1, blocks=1, clustersize=1, shmem=0)
    CUDACore.launch(dummy; threads=1, blocks=1, clustersize=1, shmem=0, stream=stream())
    CUDACore.launch(dummy; threads=1, blocks=1, clustersize=1, shmem=0, stream=stream(), cooperative=false)
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
    CUDACore.launch() do
        notify(c)
    end
    wait(c)
end

@testset "attributes" begin

md = CuModuleFile(joinpath(@__DIR__, "ptx/dummy.ptx"))
dummy = CuFunction(md, "dummy")

val = attributes(dummy)[CUDACore.FUNC_ATTRIBUTE_SHARED_SIZE_BYTES]

if CUDACore.driver_version() >= v"9.0"
    attributes(dummy)[CUDACore.FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES] = val
end

end

end

############################################################################################

@testset "graph" begin

let A = CUDACore.zeros(Int, 1)
    # ensure compilation
    A .+= 1
    @test Array(A) == [1]

    graph = capture() do
        @test is_capturing()
        A .+= 1
    end
    @test Array(A) == [1]

    exec = instantiate(graph)
    CUDACore.launch(exec)
    @test Array(A) == [2]

    graph′ = capture() do
        A .+= 2
    end

    update(exec, graph′)
    CUDACore.launch(exec)
    @test Array(A) == [4]
end

let A = CUDACore.zeros(Int, 1)
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
    a,b = CUDACore.memory_info()
    # NOTE: actually testing this is pretty fragile on CI
    #=@test a == =# CUDACore.free_memory()
    #=@test b == =# CUDACore.total_memory()
end

# dummy data
T = UInt32
N = 5
data = rand(T, N)
nb = sizeof(data)

# buffers are untyped, so we use a convenience function to get a typed pointer
# we prefer to return a device pointer (for managed buffers) to maximize CUDA coverage
typed_pointer(buf::Union{CUDACore.DeviceMemory, CUDACore.UnifiedMemory}, T) = convert(CuPtr{T}, buf)
typed_pointer(buf::CUDACore.HostMemory, T)                              = convert(Ptr{T},   buf)

@testset "showing" begin
    for (Ty, str) in zip([CUDACore.DeviceMemory, CUDACore.HostMemory, CUDACore.UnifiedMemory], ("DeviceMemory", "HostMemory", "UnifiedMemory"))
        dummy = CUDACore.alloc(Ty, 0)
        @test startswith(sprint(show, dummy), str)
        CUDACore.free(dummy)
    end
end

@testset "allocations and copies, src $srcTy dst $dstTy" for srcTy in [CUDACore.DeviceMemory, CUDACore.HostMemory, CUDACore.UnifiedMemory],
    dstTy in [CUDACore.DeviceMemory, CUDACore.HostMemory, CUDACore.UnifiedMemory]

    dummy = CUDACore.alloc(srcTy, 0)
    CUDACore.free(dummy)

    src = CUDACore.alloc(srcTy, nb)
    unsafe_copyto!(typed_pointer(src, T), pointer(data), N)

    dst = CUDACore.alloc(dstTy, nb)
    unsafe_copyto!(typed_pointer(dst, T), typed_pointer(src, T), N)

    ref = Array{T}(undef, N)
    unsafe_copyto!(pointer(ref), typed_pointer(dst, T), N)

    @test data == ref

    if isa(src, CUDACore.DeviceMemory) || isa(src, CUDACore.UnifiedMemory)
        CUDACore.memset(typed_pointer(src, T), zero(T), N)
    end

    # test the memory-type attribute
    if isa(src, CUDACore.DeviceMemory)
        @test CUDACore.memory_type(typed_pointer(src, T)) == CUDACore.MEMORYTYPE_DEVICE
    elseif isa(src, CUDACore.HostMemory)
        @test CUDACore.memory_type(convert(Ptr{T}, src)) == CUDACore.MEMORYTYPE_HOST
    elseif isa(src, CUDACore.UnifiedMemory)
        # unified memory can reside in either place
        # FIXME: does this depend on the current migration, or on the configuration?
        @test CUDACore.memory_type(convert(CuPtr{T}, src)) == CUDACore.MEMORYTYPE_HOST ||
              CUDACore.memory_type(convert(CuPtr{T}, src)) == CUDACore.MEMORYTYPE_DEVICE ||
        @test CUDACore.memory_type(convert(CuPtr{T}, src)) == CUDACore.memory_type(convert(Ptr{T}, src))
    end

    # test device with context in which pointer was allocated.
    @test device(typed_pointer(src, T)) == device()
    @test context(typed_pointer(src, T)) == context()
    if !memory_pools_supported(device())
        # NVIDIA bug #3319609
        @test context(typed_pointer(src, T)) == context()
    end

    # test the is-managed attribute
    if isa(src, CUDACore.UnifiedMemory)
        @test CUDACore.is_managed(convert(Ptr{T}, src))
        @test CUDACore.is_managed(convert(CuPtr{T}, src))
    else
        @test !CUDACore.is_managed(typed_pointer(src, T))
    end
    # Test conversion to Ptr throwing an error
    if isa(src, CUDACore.DeviceMemory)
        @test_throws ArgumentError convert(Ptr, src)
    end

    @grab_output show(stdout, src)
    @grab_output show(stdout, dst)
    CUDACore.free(src)
    CUDACore.free(dst)
end

@testset "pointer attributes" begin
    src = CUDACore.alloc(CUDACore.DeviceMemory, nb)

    attribute!(typed_pointer(src, T), CUDACore.POINTER_ATTRIBUTE_SYNC_MEMOPS, 0)

    CUDACore.free(src)
end

@testset "asynchronous operations" begin
    src = CUDACore.alloc(CUDACore.DeviceMemory, nb)

    unsafe_copyto!(typed_pointer(src, T), pointer(data), N; async=true)

    CUDACore.memset(typed_pointer(src, T), zero(T), N; stream=stream())

    CUDACore.free(src)
end

@testset "pinned memory" begin
    # create a pinned and mapped buffer
    src = CUDACore.alloc(CUDACore.HostMemory, nb, CUDACore.MEMHOSTALLOC_DEVICEMAP)

    # get the CPU address and copy some data to the buffer
    cpu_ptr = convert(Ptr{T}, src)
    @test CUDACore.memory_type(cpu_ptr) == CUDACore.MEMORYTYPE_HOST
    unsafe_copyto!(cpu_ptr, pointer(data), N)

    # get the GPU address and copy back the data
    gpu_ptr = convert(CuPtr{T}, src)
    ref = Array{T}(undef, N)
    unsafe_copyto!(pointer(ref), gpu_ptr, N)
    @test ref == data

    CUDACore.free(src)
    # NOTE: don't free dst, it's just a mapped pointer
end

# pinned memory with existing memory
if attribute(device(), CUDACore.DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED) != 0
    # register a pinned and mapped buffer
    src = CUDACore.register(CUDACore.HostMemory, pointer(data), nb, CUDACore.MEMHOSTREGISTER_DEVICEMAP)

    # get the GPU address and copy back the data
    gpu_ptr = convert(CuPtr{T}, src)
    ref = Array{T}(undef, N)
    unsafe_copyto!(pointer(ref), gpu_ptr, N)
    @test ref == data

    CUDACore.unregister(src)

    # with a RefValue
    src = Ref{T}(T(42))
    CUDACore.pin(src)
    cpu_ptr = Base.unsafe_convert(Ptr{T}, src)
    ref = Array{T}(undef, 1)
    unsafe_copyto!(pointer(ref), cpu_ptr, 1)
    @test ref == [T(42)]
end

@testset "unified memory" begin
    src = CUDACore.alloc(CUDACore.UnifiedMemory, nb)

    @test_throws BoundsError CUDACore.prefetch(src, 2*nb; device=CUDACore.DEVICE_CPU)
    # FIXME: prefetch doesn't work on some CI devices, unsure why.
    @test_skip CUDACore.prefetch(src, nb; device=CUDACore.DEVICE_CPU)
    CUDACore.advise(src, CUDACore.MEM_ADVISE_SET_READ_MOSTLY)

    # get the CPU address and copy some data
    cpu_ptr = convert(Ptr{T}, src)
    unsafe_copyto!(cpu_ptr, pointer(data), N)

    # get the GPU address and copy back data
    gpu_ptr = convert(CuPtr{T}, src)
    ref = Array{T}(undef, N)
    unsafe_copyto!(pointer(ref), gpu_ptr, N)
    @test ref == data

    CUDACore.free(src)
end

@testset "3d memcpy" begin
    # TODO: use cuMemAllocPitch (and put pitch in buffer?) to actually get benefit from this

    data = collect(reshape(1:27, 3, 3, 3))

    dst = CUDACore.alloc(CUDACore.DeviceMemory, sizeof(data))
    CUDACore.unsafe_copy3d!(typed_pointer(dst, Int), CUDACore.DeviceMemory,
                        pointer(data), CUDACore.HostMemory, length(data))

    check = zeros(Int, size(data))
    CUDACore.unsafe_copy3d!(pointer(check), CUDACore.HostMemory,
                        typed_pointer(dst, Int), CUDACore.DeviceMemory, length(data))

    @test check == data

    CUDACore.free(dst)
end
let
    # copying an x-z plane of a 3-D array

    T = Int
    nx, ny, nz = 4, 4, 4
    data = collect(reshape(1:(nx*nz), nx, nz))
    dst = CUDACore.alloc(CUDACore.DeviceMemory, nx*ny*nz*sizeof(data))

    # host to device
    CUDACore.unsafe_copy3d!(typed_pointer(dst, T), CUDACore.DeviceMemory, pointer(data), CUDACore.HostMemory,
                        nx, 1, nz;
                        dstPos=(1,2,1),
                        srcPitch=nx*sizeof(T), srcHeight=1,
                        dstPitch=nx*sizeof(T), dstHeight=ny)

    # copy back
    check = zeros(T, size(data))
    CUDACore.unsafe_copy3d!(pointer(check), CUDACore.HostMemory, typed_pointer(dst, T), CUDACore.DeviceMemory,
                        nx, 1, nz;
                        srcPos=(1,2,1),
                        srcPitch=nx*sizeof(T), srcHeight=ny,
                        dstPitch=nx*sizeof(T), dstHeight=1)

    @test check == data

    # copy back into a 3-D array
    check2 = zeros(T, nx, ny, nz)
    CUDACore.unsafe_copy3d!(pointer(check2), CUDACore.HostMemory, typed_pointer(dst, T), CUDACore.DeviceMemory,
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
    dst = CUDACore.alloc(CUDACore.DeviceMemory, nx*ny*nz*sizeof(data))

    # host to device
    CUDACore.unsafe_copy3d!(typed_pointer(dst, T), CUDACore.DeviceMemory, pointer(data), CUDACore.HostMemory,
                        1, ny, nz;
                        dstPos=(2,1,1),
                        srcPitch=1*sizeof(T), srcHeight=ny,
                        dstPitch=nx*sizeof(T), dstHeight=ny)

    # copy back
    check = zeros(T, size(data))
    CUDACore.unsafe_copy3d!(pointer(check), CUDACore.HostMemory, typed_pointer(dst, T), CUDACore.DeviceMemory,
                        1, ny, nz;
                        srcPos=(2,1,1),
                        srcPitch=nx*sizeof(T), srcHeight=ny,
                        dstPitch=1*sizeof(T), dstHeight=ny)

    @test check == data

    # copy back into a 3-D array
    check2 = zeros(T, nx, ny, nz)
    CUDACore.unsafe_copy3d!(pointer(check2), CUDACore.HostMemory, typed_pointer(dst, T), CUDACore.DeviceMemory,
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
    CUDACore.unsafe_copy3d!(pointer(A), CUDACore.HostMemory, pointer(B), CUDACore.DeviceMemory,
                        nx, 1, nz;
                        srcPos=(1,2,1),
                        srcPitch=nx*sizeof(A[1]), srcHeight=ny,
                        dstPitch=nx*sizeof(A[1]), dstHeight=1
    )

    @test A == Array(B)[:,2,:]
end

# pinned memory with existing memory
if attribute(device(), CUDACore.DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED) != 0
    hA = rand(UInt8, 512)
    @test !CUDACore.is_pinned(pointer(hA))
    CUDACore.pin(hA)
    @test CUDACore.is_pinned(pointer(hA))

    # make sure we can double-pin
    CUDACore.pin(hA)

    # memory copies on pinned memory behave differently, so test that code path
    dA = CuArray(rand(UInt8, 512))
    copyto!(dA, hA)
    copyto!(hA, dA)

    # memory copies with resized pinned memory (used to fail with CUDA_ERROR_INVALID_VALUE)
    dA = rand(Float32, 100)
    hA = Array(dA)
    @test !CUDACore.is_pinned(pointer(hA))
    for n ∈ 100:2000
        resize!(dA, n)
        resize!(hA, n)
        dA .= n
        CUDACore.pin(hA)
        @test CUDACore.is_pinned(pointer(hA))
        copyto!(hA, dA)
        copyto!(dA, hA)
    end
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

@test_throws Exception CuModule("foobar")

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
        @test_throws Exception add_data!(link, "vadd_parent", UInt8[0])
    end
end

@testset "error log" begin
    invalid_code = """
        .version 3.1
        .target sm_999
        .address_size 64"""
    @test_throws "ptxas fatal" CuModule(invalid_code)

    link = CuLink()
    @test_throws "ptxas fatal" add_data!(link, "dummy", invalid_code)
end

let
    link = CuLink()
    add_file!(link, joinpath(@__DIR__, "ptx/vadd_child.ptx"), CUDACore.JIT_INPUT_PTX)
    open(joinpath(@__DIR__, "ptx/vadd_parent.ptx")) do f
        add_data!(link, "vadd_parent", read(f, String))
    end

    obj = complete(link)
    md = CuModule(obj)

    vadd = CuFunction(md, "vadd")

    options = Dict{CUDACore.CUjit_option,Any}()
    options[CUDACore.JIT_GENERATE_LINE_INFO] = true

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
    launch_configuration(dummy; shmem=64, max_threads=typemax(Cint)+1)

    let cb_calls = 0
        launch_configuration(dummy; shmem=threads->(cb_calls += 1; 0))
        @test cb_calls > 0
    end
end

end

############################################################################################

@testset "pool" begin

dev = device()
if attribute(dev, CUDACore.DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED) == 1

pool = memory_pool(dev)

pool2 = CuMemoryPool(dev)
@test pool2 != pool
memory_pool!(dev, pool2)
@test pool2 == memory_pool(dev)
@test pool2 != default_memory_pool(dev)

memory_pool!(dev, pool)
@test pool == memory_pool(dev)

@test attribute(UInt64, pool2, CUDACore.MEMPOOL_ATTR_RELEASE_THRESHOLD) == 0
attribute!(pool2, CUDACore.MEMPOOL_ATTR_RELEASE_THRESHOLD, UInt64(2^30))
@test attribute(UInt64, pool2, CUDACore.MEMPOOL_ATTR_RELEASE_THRESHOLD) == 2^30

CUDACore.unsafe_destroy!(pool2)

end

end

############################################################################################

@testset "stream" begin

s = CuStream()
synchronize(s)
@test CUDACore.isdone(s)
@test unique_id(s) > 0

let s2 = CuStream()
    @test s != s2
    @test !(s == s2)
end

let s3 = CuStream(; flags=CUDACore.STREAM_NON_BLOCKING)
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
synchronize(stream())

@grab_output show(stdout, stream())

end

############################################################################################

@testset "version" begin

@test isa(CUDACore.driver_version(), VersionNumber)

@test isa(CUDACore.runtime_version(), VersionNumber)

end
