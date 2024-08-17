# C. Cooperative Groups

"""
CUDA.jl's cooperative groups implementation.

Cooperative groups in CUDA offer a structured approach to synchronize and communicate among
threads. They allow developers to define specific groups of threads, providing a means to
fine-tune inter-thread communication granularity. By offering a more nuanced alternative to
traditional CUDA synchronization methods, cooperative groups enable a more controlled and
efficient parallel decomposition in kernel design.

The following functionality is available in CUDA.jl:

- implicit groups: thread blocks, grid groups, and coalesced groups.
- synchronization: `sync`, `barrier_arrive`, `barrier_wait`
- warp collectives for coalesced groups: shuffle and voting
- data transfer: `memcpy_async`, `wait` and `wait_prior`

Noteworthy missing functionality:

- implicit groups: clusters, and multi-grid groups (which are deprecated)
- explicit groups: tiling and partitioning
"""
module CG

using ..CUDA
using ..CUDA: i32, Aligned, alignment

import ..LLVM
using ..LLVM.Interop
using ..LLVMLoopInfo

using Core: LLVMPtr

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
# returns something usable (e.g. a pointer to perform subsequent atomic operations)
@inline function Base.getproperty(gws::grid_workspace, sym::Symbol)
    if sym === :wsSize
        return unsafe_load(reinterpret(Ptr{UInt32}, gws))
    elseif sym === :barrier
        return reinterpret(LLVMPtr{UInt32,AS.Global}, gws) + 4
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

Returns the linearized rank of the calling thread along the interval `[1, num_threads()]`.
"""
thread_rank

"""
    num_threads(group)

Returns the total number of threads in the group.
"""
num_threads


## thread block

export this_thread_block, group_index, thread_index, dim_threads

"""
    thread_block <: thread_group

Every GPU kernel is executed by a grid of thread blocks, and threads within each block are
guaranteed to reside on the same streaming multiprocessor. A `thread_block` represents a
thread block whose dimensions are not known until runtime.

Constructed via [`this_thread_block`](@ref)
"""
struct thread_block <: thread_group
end

"""
    this_thread_block()

Constructs a `thread_block` group
"""
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

"""
    grid_group <: thread_group

Threads within this this group are guaranteed to be co-resident on the same device within
the same launched kernel. To use this group, the kernel must have been launched with `@cuda
cooperative=true`, and the device must support it (queryable device attribute).

Constructed via [`this_grid`](@ref).
"""
struct grid_group <: thread_group
    details::grid_workspace
end

"""
    this_grid()

Constructs a `grid_group`.
"""
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
    (block_rank(gg) - 1) * num_threads(tg) + thread_rank(tg)
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


## coalesced group

export coalesced_group, coalesced_threads, meta_group_rank, meta_group_size

"""
    coalesced_group <: thread_group

A group representing the current set of converged threads in a warp. The size of the group
is not guaranteed and it may return a group of only one thread (itself).

This group exposes warp-synchronous builtins. Constructed via [`coalesced_threads`](@ref).
"""
struct coalesced_group <: thread_group
    mask::UInt32
    size::UInt32
    metaGroupSize::UInt16
    metaGroupRank::UInt16

    coalesced_group(mask::UInt32) = new(mask, CUDA.popc(mask), 0, 1)
end

"""
    coalesced_threads()

Constructs a `coalesced_group`.
"""
coalesced_threads() = coalesced_group(active_mask())

num_threads(cg::coalesced_group) = cg.size

thread_rank(cg::coalesced_group) = CUDA.popc(cg.mask & CUDA.lanemask(<)) + 1i32

"""
    meta_group_rank(cg::coalesced_group)

Rank of this group in the upper level of the hierarchy.
"""
meta_group_rank(cg::coalesced_group) = cg.metaGroupRank

"""
    meta_group_size(cg::coalesced_group)

Total number of partitions created out of all CTAs when the group was created.
"""
meta_group_size(cg::coalesced_group) = cg.metaGroupSize



#
# synchronization
#

export sync, barrier_arrive, barrier_wait

"""
    sync(group)

Synchronize the threads named in the group, equivalent to calling [`barrier_wait`](@ref) and
[`barrier_arrive`](@ref) in sequence.
"""
sync

"""
    barrier_arrive(group)

Arrive on the barrier, returns a token that needs to be passed into [`barrier_wait`](@ref).
"""
barrier_arrive

"""
    barrier_wait(group, token)

Wait on the barrier, takes arrival token returned from [`barrier_arrive`](@ref).
"""
barrier_wait


## coalesced group

sync(cg::coalesced_group) = sync_warp(cg.mask)


## thread block

sync(tb::thread_block) = barrier_sync(0)

barrier_arrive(tb::thread_block) = nothing

barrier_wait(tb::thread_block, token) = barrier_sync(0)


## grid group

bar_has_flipped(oldArrive, currentArrive) = ((oldArrive ⊻ currentArrive) & 0x80000000) != 0
is_cta_master() = threadIdx().x == 1 && threadIdx().y == 1 && threadIdx().z == 1

@inline function sync(gg::grid_group)
    token = barrier_arrive(gg)
    barrier_wait(gg, token)
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
                                 Tuple{LLVMPtr{UInt32,AS.Global}, UInt32},
                                 arrived, nb)
        end
    end

    return oldArrive
end

@inline function barrier_wait(gg::grid_group, token)
    arrived = gg.details.barrier

    if is_cta_master()
        if compute_capability() < sv"7.0"
            # volatile polling; fence
            while true
                # volatile load
                current_arrive = @static if LLVM.version() >= v"17"
                    Base.llvmcall("""
                            %val = load volatile i32, ptr addrspace(1) %0
                            ret i32 %val
                        """, UInt32, Tuple{LLVMPtr{UInt32,AS.Global}}, arrived)
                else
                    Base.llvmcall("""
                            %ptr = bitcast i8 addrspace(1)* %0 to i32 addrspace(1)*
                            %val = load volatile i32, i32 addrspace(1)* %ptr
                            ret i32 %val
                        """, UInt32, Tuple{LLVMPtr{UInt32,AS.Global}}, arrived)
                end
                if bar_has_flipped(token, current_arrive)
                    break
                end
            end
            threadfence()
        else
            # polling with acquire
            while true
                current_arrive = @asmcall("ld.acquire.gpu.u32 \$0,[\$1];",
                                          "=r,l,~{memory}", true, UInt32,
                                          Tuple{LLVMPtr{UInt32,AS.Global}},
                                          arrived)
                if bar_has_flipped(token, current_arrive)
                    break
                end
            end
        end
    end

    barrier_sync(0)
end



#
# warp functions
#

## coalesced group

function shfl(cg::coalesced_group, elem, src_rank)
    lane = if src_rank == 0
        CUDA.ffs(cg.mask)
    elseif num_threads(cg) == 32
        src_rank
    else
        CUDA.fns(cg.mask, 0, src_rank) + 1i32
    end

    shfl_sync(cg.mask, elem, lane)
end

function shfl_down(cg::coalesced_group, elem, delta)
    if num_threads(cg) == 32
        return shfl_down_sync(FULL_MASK, elem, delta)
    end

    lane = CUDA.fns(cg.mask, laneid() - 1i32, delta + 1i32) + 1i32
    if lane > 32
        lane = laneid()
    end

    shfl_sync(cg.mask, elem, lane)
end

function shfl_up(cg::coalesced_group, elem, delta)
    if num_threads(cg) == 32
        return shfl_up_sync(FULL_MASK, elem, delta)
    end

    lane = CUDA.fns(cg.mask, laneid() - 1i32, -(delta + 1i32)) + 1i32
    if lane > 32
        lane = laneid()
    end

    shfl_sync(cg.mask, elem, lane)
end

vote_any(cg::coalesced_group, pred) = vote_ballot_sync(cg.mask, pred) != 0

vote_all(cg::coalesced_group, pred) = vote_all_sync(cg.mask, pred) == cg.mask

function pack_lanes(cg, laneMask)
    member_pack = UInt32(0)
    member_rank = UInt32(0)

    for bit_idx in 0:31
        lane_bit = cg.mask & (UInt32(1) << bit_idx)
        if lane_bit != 0
            if laneMask & lane_bit != 0
                member_pack |= UInt32(1) << member_rank
            end
            member_rank += 1
        end
    end

    return member_pack
end

function vote_ballot(cg::coalesced_group, pred)
    if num_threads(cg) == 32
        return vote_ballot_sync(FULL_MASK, pred)
    end

    lane_ballot = vote_ballot_sync(cg.mask, predicate)
    pack_lanes(cg, lane_ballot)
end



#
# data transfer
#


## memcpy_async API

export memcpy_async, wait, wait_prior

# TODO: thread_block_tile support (with enable_tile_optimization)
const memcpy_group = Union{thread_block, coalesced_group}

"""
    wait(group)

Make all threads in this group wait for all previously submitted [`memcpy_async`](@ref)
operations to complete.
"""
function wait(group::memcpy_group)
    wait_prior(group, 0)
    sync(group)
end

"""
    wait_prior(group, stage)

Make all threads in this group wait for all but `stage` previously submitted
[`memcpy_async`](@ref) operations to complete.
"""
function wait_prior(group::memcpy_group, stage::Integer)
    if compute_capability() >= sv"8.0"
        pipeline_wait_prior(stage)
    end
    sync(group)
end

"""
    memcpy_async(group, dst, src, bytes)

Perform a group-wide collective memory copy from `src` to `dst` of `bytes` bytes. This
operation may be performed asynchronously, so you should [`wait`](@ref) or
[`wait_prior`](@ref) before using the data. It is only supported by thread blocks and
coalesced groups.

For this operation to be performed asynchronously, the following conditions must be met:
- the source and destination memory should be aligned to 4, 8 or 16 bytes. this will be
  deduced from the datatype, but can also be specified explicitly using
  [`CUDA.align`](@ref).
- the source should be global memory, and the destination should be shared memory.
- the device should have compute capability 8.0 or higher.
"""
memcpy_async

@inline memcpy_async(group, dst, src, bytes) =
    memcpy_async(group, Aligned(dst), Aligned(src), bytes)

@inline function memcpy_async(group::memcpy_group, dst::Aligned{<:LLVMPtr},
                              src::Aligned{<:LLVMPtr}, bytes::Integer)
    _memcpy_async(group, astype(Nothing, dst[]), astype(Nothing, src[]), bytes,
                  Val(min(alignment(dst), alignment(src))))
end


## pipeline operations

pipeline_commit() = ccall("llvm.nvvm.cp.async.commit.group", llvmcall, Cvoid, ())

pipeline_wait_prior(n) =
    ccall("llvm.nvvm.cp.async.wait.group", llvmcall, Cvoid, (Int32,), n)

@generated function pipeline_memcpy_async(dst::LLVMPtr{T}, src::LLVMPtr{T}) where T
    size_and_align = sizeof(T)
    size_and_align in (4, 8, 16) || :(return error($"Unsupported size $size_and_align"))
    intr = "llvm.nvvm.cp.async.ca.shared.global.$(sizeof(T))"
    quote
        # XXX: run-time assert that dst and src are aligned
        ccall($intr, llvmcall, Cvoid,
              (LLVMPtr{T,AS.Shared}, LLVMPtr{T,AS.Global}), dst, src)
    end
end


## memcpy implementation

@inline function _memcpy_async(group, dst::LLVMPtr, src::LLVMPtr,
                               bytes, ::Val{align_hint}) where {align_hint}
    align = min(16, align_hint)
    ispow2(align) || throw(ArgumentError("Alignment must be a power of 2"))
    if compute_capability() >= sv"8.0"
        _memcpy_async_dispatch(group, Val{align}(), dst, src, bytes[])
        pipeline_commit()
    else
        inline_copy(group, astype(Int8, dst), astype(Int8, src), bytes[])
    end
end

# specializations for specific alignments, dispatching straight to aligned LDGSTS calls
@inline function _memcpy_async_dispatch(group, alignment::Val{4}, dst, src, bytes)
    T = NTuple{1, UInt32}
    src = astype(T, src)
    dst = astype(T, dst)
    accelerated_async_copy(group, dst, src, bytes ÷ sizeof(T))
end
@inline function _memcpy_async_dispatch(group, alignment::Val{8}, dst, src, bytes)
    T = NTuple{2, UInt32}
    src = astype(T, src)
    dst = astype(T, dst)
    accelerated_async_copy(group, dst, src, bytes ÷ sizeof(T))
end
@inline function _memcpy_async_dispatch(group, alignment::Val{16}, dst, src, bytes)
    T = NTuple{4, UInt32}
    src = astype(T, src)
    dst = astype(T, dst)
    accelerated_async_copy(group, dst, src, bytes ÷ sizeof(T))
end

# fallback that determines alignment at run time
@inline function _memcpy_async_dispatch(group, ::Val{align_hint}, dst, src, bytes) where {align_hint}
    alignment = find_best_alignment(dst, src, Val(align_hint), Val(16))

    # avoid copying the extra bytes
    alignment = bytes < alignment ? align_hint : alignment

    # XXX: this dispatch is weird (but it's what the STL implementation does)
    #      - we never call _memcpy_async_with_alignment because of the specializations above
    #      - the alignment is only based on the inputs, while _memcpy_async_with_alignment
    #        is intended to allow copying using a better alignment (by skipping bytes)

    if align_hint == 16
        _memcpy_async_with_alignment(group, dst, src, bytes, Val(16))
    elseif align_hint == 8
        _memcpy_async_with_alignment(group, dst, src, bytes, Val(8))
    elseif align_hint == 4
        _memcpy_async_with_alignment(group, dst, src, bytes, Val(4))
    elseif align_hint == 2
        inline_copy(group, astype(UInt16, dst), astype(UInt16, src), bytes >> 1)
    else
        inline_copy(group, astype(UInt8, dst), astype(UInt8, src), bytes)
    end
end

# force use of a specific alignment, by doing unaligned copies before and after
@inline function _memcpy_async_with_alignment(group, dst, src, bytes,
                                              ::Val{alignment}) where {alignment}
    src = astype(Int8, src)
    dst = astype(Int8, dst)

    T = NTuple{alignment ÷ sizeof(UInt32), UInt32}

    base = reinterpret(UInt64, src) % UInt32
    align_offset = ((~base) + 1i32) & (alignment - 1)

    # copy unaligned bytes
    inline_copy(group, dst, src, align_offset)
    bytes -= align_offset
    src += align_offset
    dst += align_offset

    # copy using the requested alignment
    async_count = bytes ÷ sizeof(T)
    accelerated_async_copy(group, astype(T, dst), astype(T, src), async_count)
    async_bytes = async_count * sizeof(T)

    # copy remaining unaligned bytes
    bytes -= async_bytes
    src += async_bytes
    dst += async_bytes
    inline_copy(group, dst, src, bytes)
end

@inline function accelerated_async_copy(group, dst, src, count)
    if count == 0
        return
    end

    inline_copy(group, dst, src, count)
end

@inline function accelerated_async_copy(group, dst::LLVMPtr{T,AS.Shared},
                                        src::LLVMPtr{T,AS.Global}, count) where {T}
    if count == 0
        return
    end

    if compute_capability() < sv"8.0"
        return inline_copy(group, dst, src, count)
    end

    stride = num_threads(group)
    rank = thread_rank(group)

    idx = rank
    while idx <= count
        pipeline_memcpy_async(dst + (idx - 1) * sizeof(T), src + (idx - 1) * sizeof(T))
        idx += stride
    end
end

# interleaved element by element copies from source to dest
# TODO: use alignment information to perform vectorized copies
@inline function inline_copy(group, dst, src, count)
    rank = thread_rank(group)
    stride = num_threads(group)

    idx = rank
    while idx <= count
        val = unsafe_load(src, idx)
        unsafe_store!(dst, val, idx)
        idx += stride
    end
end


## utilities

astype(::Type{T}, ptr::LLVMPtr{<:Any, AS}) where {T, AS} = reinterpret(LLVMPtr{T, AS}, ptr)

# determine best possible alignment given an input and initial conditions.
@generated function compute_num_shifts(::Val{min_alignment},
                                       ::Val{max_alignment}) where {min_alignment,
                                                                    max_alignment}
    # XXX: because of const-prop limitations with GPUCompiler.jl, force compile-time eval
    :($(floor(Int, log2(max_alignment) - log2(min_alignment))))
end
@inline function find_best_alignment(dst, src, ::Val{min_alignment},
                                     ::Val{max_alignment}) where {min_alignment,
                                                                  max_alignment}
    # calculate base addresses (we only care about the lower 32 bits)
    base1 = reinterpret(UInt64, src) % UInt32
    base2 = reinterpret(UInt64, dst) % UInt32

    # find the bits that differ (but only those that matter for the alignment check)
    diff = (base1 ⊻ base2) & (max_alignment - 1)

    # choose the best alignment (in a way that we are likely to be able to unroll)
    best = UInt32(max_alignment)
    #num_shifts = floor(Int, log2(max_alignment) - log2(min_alignment))
    num_shifts = compute_num_shifts(Val(min_alignment), Val(max_alignment))
    @loopinfo unroll for shift in num_shifts:-1:1
        alignment = UInt32(max_alignment) >> shift
        if alignment & diff != 0
            best = alignment
        end
    end
    return best
end

end

export CG


#
# deprecated
#

export this_grid, sync_grid

this_grid() = CG.this_grid()
sync_grid(grid) = CG.sync(grid)
