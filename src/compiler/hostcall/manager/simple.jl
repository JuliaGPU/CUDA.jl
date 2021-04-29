export SimpleAreaManager

"""
    SimpleAreaManager(area_count, area_size)

Simple AreaManager that makes threads fight for locks.
There are `area_count` hostcall areas.
"""
struct SimpleAreaManager <: AreaManager
    area_count::Int
    area_size::Int
end
struct SimpleData
    index::Int32
end

stride(manager::SimpleAreaManager) = align(manager.area_size + 3 * sizeof(Int64))
kind(::SimpleAreaManager) = 0
kind(::Type{SimpleAreaManager}) = 0


function Base.show(io::IO, manager::SimpleAreaManager)
    print(io, "SimpleAreaManager($(manager.area_count), $(manager.area_size))")
end

simple_meta_size() = 3 * sizeof(Int64) # lock, state, hostcall
get_simple_ptr(kind::KindConfig, data::SimpleData) = kind.area_ptr + data.index * kind.stride + simple_meta_size()

function acquire_lock_impl(::Type{SimpleAreaManager}, kind::KindConfig, hostcall::Int64)::Tuple{SimpleData, Core.LLVMPtr{Int64,AS.Global}}
    ptr = kind.area_ptr
    stride = kind.stride
    count = kind.count

    i = (blockIdx().x-1) * align(blockDim().x) + threadIdx().x - 1

    tc = 0

    cptr = ptr + (i % count) * stride
    while (!try_lock(cptr)) &&  tc < 50000
        nanosleep(UInt32(32))
        i += 1
        tc += 1
        cptr = ptr + (i % count) * stride
    end

    if tc == 50000
        @cuprintln("Timed out acquire lock")
    end

    data = SimpleData(i%count)

    return (data, get_simple_ptr(kind, data))
end


function call_host_function(kind::KindConfig, data::SimpleData, hostcall::Int64, ::Val{blocking}) where {blocking}
    cptr = kind.area_ptr + data.index * kind.stride

    ptr = reinterpret(Ptr{Int64}, cptr)
    unsafe_store!(ptr + 16, hostcall)
    unsafe_store!(ptr + 8, HOST_CALL)

    threadfence()
    notify_host(kind.notification, data.index)

    if blocking
        tc = 0
        while volatile_load(ptr + 8) != HOST_DONE &&  tc < 1000000
            nanosleep(UInt32(16))
            threadfence()
            tc += 1
        end

        if tc == 1000000
            @cuprintln("Timed out simple call")
        end

        unsafe_store!(ptr + 8, LOADING)
    else
        unlock_area(cptr)
    end
end


function finish_function(kind::KindConfig, data::SimpleData)
    cptr = kind.area_ptr + data.index * kind.stride
    unsafe_store!(cptr+8, IDLE) # challenge the gods the host did this
    unlock_area(cptr)
end

areas_in(::SimpleAreaManager, ptr::Ptr{Int64}) = Ptr{Int64}[ptr+24]
area_count(manager::SimpleAreaManager) = manager.area_count
