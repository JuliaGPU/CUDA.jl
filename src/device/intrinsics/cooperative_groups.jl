# C. Cooperative Groups

module CG

using ..CUDA
using ..CUDA: i32
using ..LLVM, ..LLVM.Interop

const cg_debug = false
if cg_debug
    cg_assert(x) = @cuassert x
    cg_abort() = @cuassert 0
else
    cg_assert(x) = return
    cg_abort() = trap()
end

function vec3_to_linear(::Type{T}, idx, dim) where T
    (T(idx.z) - one(T)) * dim.y * dim.x +
    (T(idx.y) - one(T)) * dim.x +
    T(idx.x)
end


#
# driver ABI
#

struct grid_workspace_st
    wsSize::UInt32
    barrier::UInt32
end

const grid_workspace = Ptr{grid_workspace_st}

# overload getproperty such that accessing grid workspace fields on a workspace pointer
# returns a pointer to the field (as we perform atomic operations on them)
@inline function Base.getproperty(gws::grid_workspace, sym::Symbol)
    if sym === :wsSize
        return reinterpret(Ptr{UInt32}, gws)
    elseif sym === :barrier
        return reinterpret(Ptr{UInt32}, gws) + 4
    else
        return getfield(gws, sym)
    end
end

function get_grid_workspace()
    # interpret the address from envreg 1 and 2 as the driver's grid workspace
    hi = ccall("llvm.nvvm.read.ptx.sreg.envreg1", llvmcall, UInt32, ())
    lo = ccall("llvm.nvvm.read.ptx.sreg.envreg2", llvmcall, UInt32, ())
    gridWsAbiAddress = UInt64(hi) << 32 | UInt64(lo)
    return grid_workspace(gridWsAbiAddress)
end


#
# group types
#

abstract type thread_group end

export thread_rank, num_threads

"""
    thread_rank(group)

Rank of the calling thread within [0, num_threads)
"""
thread_rank

"""
    num_threads(group)

Total number of threads in the group.
"""
num_threads


## thread block

export this_thread_block, group_index, thread_index, dim_threads

struct thread_block <: thread_group
end

this_thread_block() = thread_block()

thread_rank(tb::thread_block) = vec3_to_linear(Int32, threadIdx(), blockDim())

"""
    group_index(tb::thread_block)

3-Dimensional index of the block within the launched grid.
"""
group_index(tb::thread_block) = blockIdx()

"""
    thread_index(tb::thread_block)

3-Dimensional index of the thread within the launched block.
"""
thread_index(tb::thread_block) = threadIdx()

"""
    dim_threads(tb::thread_block)

Dimensions of the launched block in units of threads.
"""
dim_threads(tb::thread_block) = blockDim()

num_threads(tb::thread_block) = blockDim().x * blockDim().y * blockDim().z


## grid group

export grid_group, this_grid, is_valid,
       block_rank, num_blocks, dim_blocks, block_index

struct grid_group <: thread_group
    details::grid_workspace
end

@inline function this_grid()
    # load a workspace from the driver
    gg = grid_group(get_grid_workspace())
    if cg_debug
        # *all* threads must be available to synchronize
        sync(gg)
    end
    return gg
end

"""
    is_valid(gg::grid_group)

Returns whether the grid_group can synchronize
"""
is_valid(gg::grid_group) = gg.details != C_NULL

function thread_rank(gg::grid_group)
    tg = this_thread_block()
    (block_rank(gg) - 1i32) * num_threads(tg) + thread_rank(tg)
end

"""
    block_rank(gg::grid_group)

Rank of the calling block within [0, num_blocks)
"""
block_rank(gg::grid_group) = vec3_to_linear(Int64, blockIdx(), gridDim())

function num_threads(gg::grid_group)
    tg = this_thread_block()
    num_blocks(gg) * num_threads(tg)
end

"""
    num_blocks(gg::grid_group)

Total number of blocks in the group.
"""
function num_blocks(gg::grid_group)
    # y (max 65535) * z (max 65535) fits in 32 bits, so promote after multiplication
    gridDim().x * Int64(gridDim().y%UInt32 * gridDim().z%UInt)
end

"""
    dim_blocks(gg::grid_group)

Dimensions of the launched grid in units of blocks.
"""
dim_blocks(gg::grid_group) = gridDim()

"""
    block_index(gg::grid_group)

3-Dimensional index of the block within the launched grid.
"""
block_index(gg::grid_group) = blockIdx()


#
# synchronization
#

export sync, barrier_arrive, barrier_wait

"""
    sync(group)

Synchronize the threads named in the group, equivalent to `barrier_wait(barrier_arrive(g))`
"""
sync

"""
    barrier_arrive(group)

Arrive on the barrier, returns a token that needs to be passed into `barrier_wait`
"""
barrier_arrive

"""
    barrier_wait(group, token)

Wait on the barrier, takes arrival token returned from `barrier_arrive`.
"""
barrier_wait


## thread block

sync(tb::thread_block) = barrier_sync(0)

barrier_arrive(tb::thread_block) = nothing

barrier_wait(tb::thread_block, token) = barrier_sync(0)


## grid group

bar_has_flipped(oldArrive, currentArrive) = ((oldArrive ⊻ currentArrive) & 0x80000000) != 0
is_cta_master() = threadIdx().x == 1 && threadIdx().y == 1 && threadIdx().z == 1

@inline function sync(gg::grid_group)
    if !is_valid(gg)
        cg_abort()
    end
    arrived = gg.details.barrier

    barrier_sync(0)

    if is_cta_master()
        expected = gridDim().x * gridDim().y * gridDim().z
        gpu_master = blockIdx().x == 1 && blockIdx().y == 1 && blockIdx().z == 1

        nb = UInt32(1)
        if gpu_master
            nb = 0x80000000 - (expected - UInt32(1))
        end

        if compute_capability() < sv"7"
            # fence; barrier update; volatile polling; fence
            threadfence()

            oldArrive = CUDA.atomic_add!(arrived, nb)

            while !bar_has_flipped(oldArrive, unsafe_load(arrived))
                # spin
            end

            threadfence()
        else
            # barrier update with release; polling with acquire
            oldArrive = @asmcall("atom.add.release.gpu.u32 \$0,[\$1],\$2;",
                                 "=r,l,r,~{memory}", true, UInt32,
                                 Tuple{Ptr{UInt32}, UInt32}, arrived, nb)

            while true
                current_arrive = @asmcall("ld.acquire.gpu.u32 \$0,[\$1];",
                                          "=r,l,~{memory}", true, UInt32,
                                          Tuple{Ptr{UInt32}}, arrived)
                if bar_has_flipped(oldArrive, current_arrive)
                    break
                end
            end
        end
    end

    barrier_sync(0)
end

@inline function barrier_arrive(gg::grid_group)
    if !is_valid(gg)
        cg_abort()
    end
    arrived = gg.details.barrier
    oldArrive = UInt32(0)

    barrier_sync(0)

    if is_cta_master()
        expected = gridDim().x * gridDim().y * gridDim().z
        gpu_master = blockIdx().x == 1 && blockIdx().y == 1 && blockIdx().z == 1

        nb = UInt32(1)
        if gpu_master
            nb = 0x80000000 - (expected - UInt32(1))
        end

        if compute_capability() < sv"7.0"
            # fence; barrier update
            threadfence()

            oldArrive = CUDA.atomic_add!(arrived, nb)
        else
            # barrier update with release
            oldArrive = @asmcall("atom.add.release.gpu.u32 \$0,[\$1],\$2;",
                                 "=r,l,r,~{memory}", true, UInt32,
                                 Tuple{Ptr{UInt32}, UInt32}, arrived, nb)
        end
    end

    return oldArrive
end

@inline function barrier_wait(gg::grid_group, token)
    arrived = gg.details.barrier

    if is_cta_master()
        if compute_capability() < sv"7.0"
            # volatile polling; fence
            while !bar_has_flipped(token, unsafe_load(arrived))
                # spin
            end
            threadfence()
        else
            # polling with acquire
            while true
                current_arrive = @asmcall("ld.acquire.gpu.u32 \$0,[\$1];",
                                          "=r,l,~{memory}", true, UInt32,
                                          Tuple{Ptr{UInt32}}, arrived)
                if bar_has_flipped(token, current_arrive)
                    break
                end
            end
        end
    end

    barrier_sync(0)
end

end

export CG


#
# deprecated
#

export this_grid, sync_grid

this_grid() = CG.this_grid()
sync_grid(grid) = CG.sync(grid)
