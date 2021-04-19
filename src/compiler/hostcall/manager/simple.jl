"""
    SimpleAreaManager(area_count, area_size)

Simple AreaManager that makes threads fight for locks.
There are `area_count` hostcall areas.
"""
struct SimpleAreaManager <: AreaManager
    area_count::Int
    area_size::Int
end
stride(manager::SimpleAreaManager) = align(manager.area_size + 3 * sizeof(Int64))
kind(::SimpleAreaManager) = 0
kind(::Type{SimpleAreaManager}) = 0

get_simple_ptr(kind::KindConfig, data::Data) = kind.area_ptr + data.a * kind.stride + 3 * sizeof(Int64)

function acquire_lock_impl(::Type{SimpleAreaManager}, kind::KindConfig, hostcall::Int64, blocking::Val{B})::Tuple{Data, Core.LLVMPtr{Int64,AS.Global}} where {B}
    ptr = kind.area_ptr
    stride = kind.stride
    count = kind.count

    i = threadIdx().x - 1

    tc = 0

    cptr = ptr + (i % count) * stride
    while (!try_lock(cptr)) &&  tc < 80000
        nanosleep(UInt32(16))
        i += 1
        tc += 1
        cptr = ptr + (i % count) * stride
    end

    if tc == 80000
        @cuprintln("Timed out")
    end

    data = Data(i%count,0, 0, 0)

    return (data, get_simple_ptr(kind, data))
end


function call_host_function_impl(::Type{SimpleAreaManager}, kind::KindConfig, data::Data, hostcall::Int64, ::Val{true})
    index = data.a
    cptr = kind.area_ptr + index * kind.stride

    ptr = reinterpret(Ptr{Int64}, cptr)
    unsafe_store!(ptr + 16, hostcall)
    unsafe_store!(ptr + 8, HOST_CALL_BLOCKING)
    threadfence()

    while volatile_load(ptr + 8) != HOST_DONE
        nanosleep(UInt32(16))
        threadfence()
    end

    unsafe_store!(ptr + 8, LOADING)
end

function call_host_function_impl(::Type{SimpleAreaManager}, kind::KindConfig, data::Data, hostcall::Int64, ::Val{false})
    index = data.a
    cptr = kind.area_ptr + index * kind.stride

    unsafe_store!(cptr + 16, hostcall)
    unsafe_store!(cptr + 8, HOST_CALL_NON_BLOCKING)
    threadfence()

    unlock_area(cptr)
end


function finish_function_impl(::Type{SimpleAreaManager}, kind::KindConfig, data::Data)
    index = data.a
    cptr = kind.area_ptr + index * kind.stride
    unsafe_store!(cptr+8, IDLE) # challenge the gods the host did this
    unlock_area(cptr)
end

areas_in(::SimpleAreaManager, ptr::Ptr{Int64}) = Ptr{Int64}[ptr+24]
area_count(manager::SimpleAreaManager) = manager.area_count
