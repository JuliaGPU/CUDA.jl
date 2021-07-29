export WarpAreaManager

"""
    WarpAreaManager(warp_area_count, area_size, warp_size=32)

Less simple AreaManager that locks an area per warp when invoked, reducing congestion.
Uses Opportunistic Warp-level Programming to achieve this.
"""
struct WarpAreaManager <: AreaManager
    warp_area_count::Int
    area_size::Int
    warp_size::Int
end
WarpAreaManager(warp_area_count, area_size) = WarpAreaManager(warp_area_count, area_size, 32)

struct WarpData
    index::Int32
    mask::UInt32
    laneid::Int32
    leader::Int32
end
const WARP_META_SIZE = 4 * sizeof(Int64)# lock, state, hostcall, mask

meta_size(::WarpAreaManager) = WARP_META_SIZE
stride(manager::WarpAreaManager) = align(manager.area_size * manager.warp_size)
kind(::WarpAreaManager) = 1
kind(::Type{WarpAreaManager}) = 1
area_count(manager::WarpAreaManager) = manager.warp_area_count


function Base.show(io::IO, manager::WarpAreaManager)
    print(io, "WarpAreaManager($(manager.warp_area_count), $(manager.area_size))")
end


get_warp_ptr(kind::KindConfig, data::WarpData) = kind.area_ptr + data.index * kind.stride + (data.laneid - 1) * kind.area_size

function acquire_lock_impl(::Type{WarpAreaManager}, kind::KindConfig, hostcall::Int64)::Tuple{WarpData, Core.LLVMPtr{Int64,AS.Global}}
    mask1 = vote_ballot(true)
    leader = _ffs(mask1) # determine first 1 in mask, that's our true leader!
    threadx = (blockIdx().x-1) * align(blockDim().x) + threadIdx().x - 1
    laneid = (threadx % 32) + 1

    leader_hostcall = shfl_sync(mask1, hostcall, leader)
    mask = vote_ballot_sync(mask1, leader_hostcall == hostcall)

    if hostcall != leader_hostcall
        return (WarpData(0,0,0,0), 0)
    end

    index = 0
    if leader == laneid
        # acquire lock for warp
        ptr = kind.meta_ptr
        count = kind.count

        i = div(threadx, ws) % count

        tc = 0

        cptr = ptr + i * WARP_META_SIZE
        while(!try_lock(cptr)) && tc < 1000000
            nanosleep(UInt32(32))
            i += 1
            tc += 1
            cptr += WARP_META_SIZE
            if i == count
                i = 0
                cptr = ptr
            end
        end

        if tc == 1000000
            @cuprintln("Timed out acquire lock")
        end


        index = i
    end

    index = shfl_sync(mask, index, leader)

    data = WarpData(index, mask, laneid, leader)
    ptr = get_warp_ptr(kind, data)

    (data, ptr)
end


function call_host_function(kind::KindConfig, data::WarpData, hostcall::Int64, ::Val{blocking}) where {blocking}
    if data.laneid == data.leader
        cptr = kind.meta_ptr + data.index * WARP_META_SIZE
        ptr = reinterpret(Ptr{Int64}, cptr)
        unsafe_store!(ptr + 16, hostcall)
        unsafe_store!(ptr + 24, data.mask)
        unsafe_store!(ptr+8, HOST_CALL)

        threadfence()
        notify_host(kind.notification, data.index)

        if blocking
            tc = 0
            while volatile_load(ptr + 16) != 0 && tc < 500000
                nanosleep(UInt32(16))
                tc += 1
                threadfence()
            end

            if tc == 500000
                @cuprintln("Warp timed out")
            end

            unsafe_store!(ptr + 8, LOADING)

        else
            unlock_area(cptr)
        end
    end

    sync_warp(data.mask)
end



function finish_function(kind::KindConfig, data::WarpData)
    if data.laneid == data.leader # leader
        cptr = kind.meta_ptr + data.index * WARP_META_SIZE

        unsafe_store!(cptr+24, 0)
        unsafe_store!(cptr+8, IDLE)

        unlock_area(cptr)
    end
    sync_warp(data.mask)
end


function areas_in(manager::WarpAreaManager, meta_ptr::Ptr{Int64}, area_ptr::Ptr{Int64})
    ptrs = Ptr{Int64}[]
    mask = unsafe_load(meta_ptr + 24)
    st = manager.area_size
    for digit in digits(mask, base=2, pad=32)
        if digit == 1
            push!(ptrs, area_ptr)
        end
        area_ptr += st
    end

    return ptrs
end
