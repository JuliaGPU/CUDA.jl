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

struct WarpData
    index::Int32
    mask::UInt32
    laneid::Int32
    leader::Int32
end

warp_meta_size() = 4 * sizeof(Int64) # lock, state, hostcall, mask
stride(manager::WarpAreaManager) = align(manager.area_size * manager.warp_size + warp_meta_size())
kind(::WarpAreaManager) = 1
kind(::Type{WarpAreaManager}) = 1
area_count(manager::WarpAreaManager) = manager.warp_area_count


get_warp_ptr(kind::KindConfig, data::WarpData) = kind.area_ptr + data.index * kind.stride + warp_meta_size() + (data.laneid - 1) * kind.area_size

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
        i = div(threadx, ws)

        ptr = kind.area_ptr
        stride = kind.stride
        count = kind.count

        tc = 0
        cptr = ptr + (i % count) * stride
        while(!try_lock(cptr)) && tc < 50000
            nanosleep(UInt32(32))
            i += 1
            tc += 1
            cptr = ptr + (i % count) * stride
        end

        if tc == 50000
            @cuprintln("Timed out")
        end


        index = i % count
    end

    index = shfl_sync(mask, index, leader)

    data = WarpData(index, mask, laneid, leader)
    ptr = get_warp_ptr(kind, data)

    (data, ptr)
end


function call_host_function(kind::KindConfig, data::WarpData, hostcall::Int64, ::Val{true})
    if data.laneid == data.leader
        ptr = reinterpret(Ptr{Int64}, kind.area_ptr + data.index * kind.stride)
        unsafe_store!(ptr + 16, hostcall)
        unsafe_store!(ptr + 24, data.mask)

        threadfence()
        unsafe_store!(ptr+8, HOST_CALL_BLOCKING)

        while volatile_load(ptr + 16) != 0
            nanosleep(UInt32(16))
            threadfence()
        end

        unsafe_store!(ptr + 8, LOADING)
    end

    sync_warp(data.mask)
end


function call_host_function(kind::KindConfig, data::WarpData, hostcall::Int64, ::Val{false})
    (index, mask, laneid, leader) = warp_destruct_data(data)

    if data.laneid == data.leader
        cptr = kind.area_ptr + data.index * kind.stride
        unsafe_store!(cptr + 16, hostcall)
        unsafe_store!(cptr + 24, data.mask)

        threadfence()
        unsafe_store!(cptr+8, HOST_CALL_NON_BLOCKING)

        unlock_area(cptr)
    end

    sync_warp(data.mask)
end


function finish_function(kind::KindConfig, data::WarpData)
    if data.laneid == data.leader # leader
        cptr = kind.area_ptr + data.index * kind.stride

        unsafe_store!(cptr+24, 0)
        unsafe_store!(cptr+16, IDLE)

        unlock_area(cptr)
    end
end


function areas_in(manager::WarpAreaManager, ptr::Ptr{Int64})
    ptrs = Ptr{Int64}[]
    mask = unsafe_load(ptr + 24)
    ptr += warp_meta_size()
    st = manager.area_size
    for digit in digits(mask, base=2, pad=32)
        if digit == 1
            push!(ptrs, ptr)
        end
        ptr += st
    end

    return ptrs
end
