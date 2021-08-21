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
    @test CuDevice(typed_pointer(src, T)) == device()
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
    # can only get GPU pointer if the pinned buffer is mapped
    src = Mem.alloc(Mem.Host, nb)
    @test_throws ArgumentError convert(CuPtr{T}, src)
    Mem.free(src)

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
    # can only get GPU pointer if the pinned buffer is mapped
    @test_throws ArgumentError Mem.register(Mem.Host, pointer(data), 0)
    src = Mem.register(Mem.Host, pointer(data), nb)
    @test_throws ArgumentError convert(CuPtr{T}, src)
    Mem.unregister(src)

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
