"""
    SimpleAreaManager(area_count, area_size)

Simple AreaManager that makes threads fight for locks.
There are `area_count` hostcall areas.
"""
struct SimpleAreaManager <: AreaManager
    area_count::Int
    area_size::Int
end
stride(manager::SimpleAreaManager) = align(manager.area_size + 2 * sizeof(Int64))
kind(::SimpleAreaManager) = 0
kind(::Type{SimpleAreaManager}) = 0

get_simple_ptr(kind::KindConfig, data::Data) = kind.area_ptr + data.a * kind.stride + 16

function acquire_lock_impl(::Type{SimpleAreaManager}, kind::KindConfig, hostcall::Int64)::Tuple{Data, Core.LLVMPtr{Int64,AS.Global}}
    ptr = kind.area_ptr
    stride = kind.stride
    count = kind.count

    i = threadIdx().x - 1

    while atomic_cas!(ptr + (i % count) * stride, IDLE, LOADING) != IDLE
        nanosleep(UInt32(16))
        i += 1
    end

    data = Data(i%count,0, 0, 0)

    return (data, get_simple_ptr(kind, data))
end

function call_host_function_impl(::Type{SimpleAreaManager}, kind::KindConfig, data::Data, hostcall::Int64)
    index = data.a

    ptr = reinterpret(Ptr{Int64}, kind.area_ptr + index * kind.stride)
    unsafe_store!(ptr + 8, hostcall)
    threadfence()
    unsafe_store!(ptr, HOST_CALL)

    while volatile_load(ptr + 8) != 0
        nanosleep(UInt32(16))
        threadfence()
    end

    unsafe_store!(ptr, LOADING)
end

function finish_function_impl(::Type{SimpleAreaManager}, kind::KindConfig, data::Data)
    index = data.a

    ptr = get_simple_ptr(kind, data) - 16 # get base pointer
    unsafe_store!(ptr+16, 0)
    unsafe_store!(ptr+8, 0)

    unsafe_store!(ptr, IDLE)
end

areas_in(::SimpleAreaManager, ptr::Ptr{Int64}) = Ptr{Int64}[ptr+16]
area_count(manager::SimpleAreaManager) = manager.area_count
