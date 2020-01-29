@testset "memory" begin

let
    a,b = Mem.info()
    # NOTE: actually testing this is pretty fragile on CI
    #=@test a == =# CUDAdrv.available_memory()
    #=@test b == =# CUDAdrv.total_memory()
end

# dummy data
T = UInt32
N = 5
data = rand(T, N)
nb = sizeof(data)

# buffers are untyped, so we use a convenience function to get a typed pointer
# we prefer to return a device pointer (for managed buffers) to maximize CUDAdrv coverage
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
        @test CUDAdrv.memory_type(typed_pointer(src, T)) == CUDAdrv.MEMORYTYPE_DEVICE
    elseif isa(src, Mem.Host)
        @test CUDAdrv.memory_type(convert(Ptr{T}, src)) == CUDAdrv.MEMORYTYPE_HOST
    elseif isa(src, Mem.Unified)
        # unified memory can reside in either place
        # FIXME: does this depend on the current migration, or on the configuration?
        @test CUDAdrv.memory_type(convert(CuPtr{T}, src)) == CUDAdrv.MEMORYTYPE_HOST ||
              CUDAdrv.memory_type(convert(CuPtr{T}, src)) == CUDAdrv.MEMORYTYPE_DEVICE ||
        @test CUDAdrv.memory_type(convert(CuPtr{T}, src)) == CUDAdrv.memory_type(convert(Ptr{T}, src))
    end

    # test the is-managed attribute
    if isa(src, Mem.Unified)
        @test CUDAdrv.is_managed(convert(Ptr{T}, src))
        @test CUDAdrv.is_managed(convert(CuPtr{T}, src))
    else
        @test !CUDAdrv.is_managed(typed_pointer(src, T))
    end

    Mem.free(src)
    Mem.free(dst)
end

# pointer attributes
let
    src = Mem.alloc(Mem.Device, nb)

    attribute!(typed_pointer(src, T), CUDAdrv.POINTER_ATTRIBUTE_SYNC_MEMOPS, 0)

    Mem.free(src)
end

# asynchronous operations
let
    src = Mem.alloc(Mem.Device, nb)

    @test_throws ArgumentError unsafe_copyto!(typed_pointer(src, T), pointer(data), N; async=true)
    unsafe_copyto!(typed_pointer(src, T), pointer(data), N; async=true, stream=CuDefaultStream())

    Mem.set!(typed_pointer(src, T), zero(T), N; async=true, stream=CuDefaultStream())

    Mem.free(src)
end

# pinned memory
let
    # can only get GPU pointer if the pinned buffer is mapped
    src = Mem.alloc(Mem.Host, nb)
    @test_throws ArgumentError convert(CuPtr{T}, src)
    Mem.free(src)

    # create a pinned and mapped buffer
    src = Mem.alloc(Mem.Host, nb, Mem.HOSTALLOC_DEVICEMAP)

    # get the CPU address and copy some data
    cpu_ptr = convert(Ptr{T}, src)
    @test CUDAdrv.memory_type(cpu_ptr) == CUDAdrv.MEMORYTYPE_HOST
    unsafe_copyto!(cpu_ptr, pointer(data), N)

    # get the GPU address and construct a fake device buffer
    gpu_ptr = convert(CuPtr{Cvoid}, src)
    @test CUDAdrv.memory_type(gpu_ptr) == CUDAdrv.MEMORYTYPE_HOST
    gpu_obj = Mem.alloc(Mem.Device, nb)
    dst = similar(gpu_obj, gpu_ptr)
    Mem.free(gpu_obj)

    # copy data back from the GPU and compare
    ref = Array{T}(undef, N)
    unsafe_copyto!(pointer(ref), typed_pointer(dst, T), N)
    @test ref == data

    Mem.free(src)
    # NOTE: don't free dst, it's just a mapped pointer
end

# pinned memory with existing memory
if attribute(dev, CUDAdrv.DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED) != 0
    # can only get GPU pointer if the pinned buffer is mapped
    @test_throws ArgumentError Mem.register(Mem.Host, pointer(data), 0)
    src = Mem.register(Mem.Host, pointer(data), nb)
    @test_throws ArgumentError convert(CuPtr{T}, src)
    Mem.unregister(src)

    # register a pinned and mapped buffer
    src = Mem.register(Mem.Host, pointer(data), nb, Mem.HOSTREGISTER_DEVICEMAP)

    # get the GPU address and construct a fake device buffer
    gpu_ptr = convert(CuPtr{Cvoid}, src)
    gpu_obj = Mem.alloc(Mem.Device, nb)
    dst = similar(gpu_obj, gpu_ptr)
    Mem.free(gpu_obj)

    # copy data back from the GPU and compare
    ref = Array{T}(undef, N)
    unsafe_copyto!(pointer(ref), typed_pointer(dst, T), N)
    @test ref == data

    Mem.unregister(src)
    # NOTE: don't unregister dst, it's just a mapped pointer
end

# unified memory
let
    src = Mem.alloc(Mem.Unified, nb)

    #Mem.prefetch(src, nb; device=CUDAdrv.DEVICE_CPU)
    Mem.advise(src, Mem.ADVISE_SET_READ_MOSTLY)

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
    unsafe_copyto!(pointer(ref), typed_pointer(dst, T), N)
    @test ref == data

    Mem.free(src)
end

end
