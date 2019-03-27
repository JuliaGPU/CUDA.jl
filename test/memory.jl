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

    src = Mem.alloc(srcTy, nb)
    Mem.copy!(src, pointer(data), nb)

    dst = Mem.alloc(dstTy, nb)
    Mem.copy!(dst, src, nb)

    ref = Array{T}(undef, N)
    Mem.copy!(pointer(ref), dst, nb)

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
    src = Mem.alloc(Mem.Host, nb, Mem.HOSTALLOC_DEVICEMAP)

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
    Mem.copy!(pointer(ref), dst, nb)
    @test ref == data

    Mem.free(src)
end

end
