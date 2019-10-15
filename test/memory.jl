@testset "memory" begin

let
    a,b = Mem.info()
    # NOTE: actually testing this is pretty fragile on CI
    #=@test a == =# CUDAdrv.available_memory()
    #=@test b == =# CUDAdrv.total_memory()
end

# dummy data
T = Int
N = 5
data = rand(T, N)
nb = sizeof(data)

# allocations and copies
for srcTy in [Mem.Device, Mem.Host, Mem.Unified],
    dstTy in [Mem.Device, Mem.Host, Mem.Unified]

    dummy = Mem.alloc(srcTy, 0)
    Mem.free(dummy)

    src = Mem.alloc(srcTy, nb)
    if isa(src, Mem.Host)
        unsafe_copyto!(convert(Ptr{T}, src), pointer(data), N)
    else
        Mem.copy!(src, pointer(data), nb)
    end

    dst = Mem.alloc(dstTy, nb)
    if isa(src, Mem.Host) && isa(dst, Mem.Host)
        unsafe_copyto!(convert(Ptr{T}, dst), convert(Ptr{T}, src), N)
    else
        Mem.copy!(dst, src, nb)
    end

    ref = Array{T}(undef, N)
    if isa(dst, Mem.Host)
        unsafe_copyto!(pointer(ref), convert(Ptr{T}, dst), N)
    else
        Mem.copy!(pointer(ref), dst, nb)
    end

    @test data == ref

    Mem.free(src)
    Mem.free(dst)
end

# asynchronous operations
let
    src = Mem.alloc(Mem.Device, nb)
    @test_throws ArgumentError Mem.copy!(src, pointer(data), nb; async=true)
    Mem.copy!(src, pointer(data), nb; async=true, stream=CuDefaultStream())

    Mem.free(src)
end

# pinned memory
let
    # can only get GPU pointer if the pinned buffer is mapped
    src = Mem.alloc(Mem.Host, nb)
    @test_throws ArgumentError convert(CuPtr{T}, src)
    Mem.free(src)

    # create a pinned and mapped buffer
    src = Mem.alloc(Mem.Host, nb, CUDAdrv.MEMHOSTALLOC_DEVICEMAP)

    # get the CPU address and copy some data
    cpu_ptr = convert(Ptr{T}, src)
    unsafe_copyto!(cpu_ptr, pointer(data), N)

    # get the GPU address and construct a fake device buffer
    gpu_ptr = convert(CuPtr{Cvoid}, src)
    gpu_obj = Mem.alloc(Mem.Device, nb)
    dst = similar(gpu_obj, gpu_ptr)
    Mem.free(gpu_obj)

    # copy data back from the GPU and compare
    ref = Array{T}(undef, N)
    Mem.copy!(pointer(ref), dst, nb)
    @test ref == data

    Mem.free(src)
    # NOTE: don't free dst, it's just a mapped pointer
end

# pinned memory with existing memory
let
    # can only get GPU pointer if the pinned buffer is mapped
    src = Mem.register(Mem.Host, pointer(data), nb)
    @test_throws ArgumentError convert(CuPtr{T}, src)
    Mem.unregister(src)

    # register a pinned and mapped buffer
    src = Mem.register(Mem.Host, pointer(data), nb, CUDAdrv.MEMHOSTREGISTER_DEVICEMAP)

    # get the GPU address and construct a fake device buffer
    gpu_ptr = convert(CuPtr{Cvoid}, src)
    gpu_obj = Mem.alloc(Mem.Device, nb)
    dst = similar(gpu_obj, gpu_ptr)
    Mem.free(gpu_obj)

    # copy data back from the GPU and compare
    ref = Array{T}(undef, N)
    Mem.copy!(pointer(ref), dst, nb)
    @test ref == data

    Mem.unregister(src)
    # NOTE: don't unregister dst, it's just a mapped pointer
end

# unified memory
let
    src = Mem.alloc(Mem.Unified, nb)

    #Mem.prefetch(src, nb; device=CUDAdrv.DEVICE_CPU)
    Mem.advise(src, CUDAdrv.MEM_ADVISE_SET_READ_MOSTLY)

    # get the CPU address and copy some data
    cpu_ptr = convert(Ptr{T}, src)
    unsafe_copyto!(cpu_ptr, pointer(data), N)

    # get the GPU address and construct a fake device buffer
    gpu_ptr = convert(CuPtr{Cvoid}, src)
    gpu_obj = Mem.alloc(Mem.Device, nb)
    dst = similar(gpu_obj, gpu_ptr)
    Mem.free(gpu_obj)

    # copy data back from the GPU and compare
    ref = Array{T}(undef, N)
    Mem.copy!(pointer(ref), dst, nb)
    @test ref == data

    Mem.free(src)
end

end
