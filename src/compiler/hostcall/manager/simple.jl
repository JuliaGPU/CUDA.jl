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
const SIMPLE_META_SIZE = 3 * sizeof(Int64)
meta_size(::SimpleAreaManager) = SIMPLE_META_SIZE

stride(manager::SimpleAreaManager) = align(manager.area_size)
kind(::SimpleAreaManager) = 0
kind(::Type{SimpleAreaManager}) = 0


function Base.show(io::IO, manager::SimpleAreaManager)
    print(io, "SimpleAreaManager($(manager.area_count), $(manager.area_size))")
end


get_simple_ptr(kind::KindConfig, data::SimpleData) = kind.area_ptr + data.index * kind.stride


function acquire_lock_impl(::Type{SimpleAreaManager}, kind::KindConfig, hostcall::Int64)::Tuple{SimpleData, Core.LLVMPtr{Int64,AS.Global}}
    ptr = kind.meta_ptr
    count = kind.count

    i = ((blockIdx().x-1) * align(blockDim().x) + threadIdx().x - 1) % count

    tc = 0

    cptr = ptr + i * SIMPLE_META_SIZE
    while (!try_lock(cptr)) &&  tc < 250000
        nanosleep(UInt32(32))
        i += 1
        tc += 1

        cptr += SIMPLE_META_SIZE
        if i == count
            i = 0
            cptr = ptr
        end
    end

    if tc == 250000
        @cuprintln("Timed out acquire lock")
    end

    data = SimpleData(i%count)

    return (data, get_simple_ptr(kind, data))
end


function call_host_function(kind::KindConfig, data::SimpleData, hostcall::Int64, ::Val{blocking}) where {blocking}
    # Setup meta data for function call
    cptr = kind.meta_ptr + data.index * SIMPLE_META_SIZE


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
    cptr = kind.meta_ptr + data.index * SIMPLE_META_SIZE
    unsafe_store!(cptr+8, IDLE) # challenge the gods the host did this
    unlock_area(cptr)
end

areas_in(::SimpleAreaManager, ::Ptr{Int64}, area_ptr::Ptr{Int64}) = Ptr{Int64}[area_ptr]
area_count(manager::SimpleAreaManager) = manager.area_count
