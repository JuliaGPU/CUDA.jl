# Synchronization (B.6)

export sync_threads, sync_warp
export sync_threads_count, sync_threads_and, sync_threads_or
export threadfence, threadfence_block, threadfence_system

"""
    sync_threads()

Waits until all threads in the thread block have reached this point and all global and
shared memory accesses made by these threads prior to `sync_threads()` are visible to all
threads in the block.
"""
@inline sync_threads() = ccall("llvm.nvvm.barrier0", llvmcall, Cvoid, ())

"""
    sync_threads_count(predicate::Int32)

Identical to `__syncthreads()` with the additional feature that it evaluates predicate for
all threads of the block and returns the number of threads for which `predicate` evaluates
to non-zero.
"""
@inline sync_threads_count(predicate::Int32) = ccall("llvm.nvvm.barrier0.popc", llvmcall, Int32, (Int32,), predicate)

"""
    sync_threads_count(predicate::Bool)

Identical to `__syncthreads()` with the additional feature that it evaluates predicate for
all threads of the block and returns the number of threads for which `predicate` evaluates
to `true`.
"""
@inline sync_threads_count(predicate::Bool) = sync_threads_count(Int32(predicate))

"""
    sync_threads_and(predicate::Int32)

Identical to `__syncthreads()` with the additional feature that it evaluates predicate for
all threads of the block and returns non-zero if and only if `predicate` evaluates to
non-zero for all of them.
"""
@inline sync_threads_and(predicate::Int32) = ccall("llvm.nvvm.barrier0.and", llvmcall, Int32, (Int32,), predicate)

"""
    sync_threads_and(predicate::Bool)

Identical to `__syncthreads()` with the additional feature that it evaluates predicate for
all threads of the block and returns `true` if and only if `predicate` evaluates to `true`
for all of them.
"""
@inline sync_threads_and(predicate::Bool) = ifelse(sync_threads_and(Int32(predicate)) != Int32(0), true, false)

"""
    sync_threads_or(predicate::Int32)

Identical to `__syncthreads()` with the additional feature that it evaluates predicate for
all threads of the block and returns non-zero if and only if `predicate` evaluates to
non-zero for any of them.
"""
@inline sync_threads_or(predicate::Int32) = ccall("llvm.nvvm.barrier0.or", llvmcall, Int32, (Int32,), predicate)

"""
    sync_threads_or(predicate::Bool)

Identical to `__syncthreads()` with the additional feature that it evaluates predicate for
all threads of the block and returns `true` if and only if `predicate` evaluates to `true`
for any of them.
"""
@inline sync_threads_or(predicate::Bool) = ifelse(sync_threads_or(Int32(predicate)) != Int32(0), true, false)

"""
    sync_warp(mask::Integer=0xffffffff)

Waits threads in the warp, selected by means of the bitmask `mask`, have reached this point
and all global and shared memory accesses made by these threads prior to `sync_warp()` are
visible to those threads in the warp. The default value for `mask` selects all threads in
the warp.

!!! note
    Requires CUDA >= 9.0 and sm_6.2
"""
@inline function sync_warp(mask::Integer=0xffffffff)
    @asmcall("bar.warp.sync \$0;", "r", true,
             Cvoid, Tuple{UInt32}, convert(UInt32, mask))
end

@inline threadfence(::BlockScope) = threadfence_block()
@inline threadfence_block() = ccall("llvm.nvvm.membar.cta", llvmcall, Cvoid, ())
@inline threadfence_sc_block() = @asmcall("fence.sc.cta;", "~{memory}", true, Cvoid, Tuple{})
@inline threadfence_acq_rel_block() = @asmcall("fence.acq_rel.cta;", "~{memory}", true, Cvoid, Tuple{})

function atomic_thread_fence(order, scope::BlockScope)
    if compute_capability() >= sv"7.0"
        if order == seq_cst
            threadfence_sc_block()
        elseif order == acquire || order == acq_rel || order == release # || order == consume
            threadfence_acq_rel_block()
        else
            throw(AtomicOrderUnsupported(order))
        end
    else
        if order == seq_cst ||
         # order == consume ||
           order == acquire ||
           order == acq_rel ||
           order == release

            threadfence_block()
        else
            throw(AtomicOrderUnsupported(order))
        end
    end
end

@inline threadfence(::DeviceScope=device_scope) = threadfence_device()
@inline threadfence_device() = ccall("llvm.nvvm.membar.gl", llvmcall, Cvoid, ())
@inline threadfence_sc_device() = @asmcall("fence.sc.gpu;", "~{memory}", true, Cvoid, Tuple{})
@inline threadfence_acq_rel_device() = @asmcall("fence.acq_rel.gpu;", "~{memory}", true, Cvoid, Tuple{})

function atomic_thread_fence(order, scope::DeviceScope=device_scope)
    if compute_capability() >= sv"7.0"
        if order == seq_cst

            threadfence_sc_device()
        elseif order == acquire ||
             # order == consume ||
               order == acq_rel ||
               order == release

            threadfence_acq_rel_device()
        else
            throw(AtomicOrderUnsupported(order))
        end
    else
        if order == seq_cst ||
           order == consume ||
           order == acquire ||
           order == acq_rel ||
           order == release

            threadfence_device()
        else
            throw(AtomicOrderUnsupported(order))
        end
    end
end

@inline threadfence(::SystemScope) = threadfence_system()
@inline threadfence_system() = ccall("llvm.nvvm.membar.sys", llvmcall, Cvoid, ())
@inline threadfence_sc_system() = @asmcall("fence.sc.sys;", "~{memory}", true, Cvoid, Tuple{})
@inline threadfence_acq_rel_system() = @asmcall("fence.acq_rel.sys;", "~{memory}", true, Cvoid, Tuple{})

function atomic_thread_fence(order, scope::SystemScope)
    if compute_capability() >= sv"7.0"
        if order == seq_cst

            threadfence_sc_system()
        elseif order == acquire ||
            #  order == consume ||
               order == acq_rel ||
               order == release

            threadfence_acq_rel_system()
        else
            throw(AtomicOrderUnsupported(order))
        end
    else
        if order == seq_cst ||
         # order == consume ||
           order == acquire ||
           order == acq_rel ||
           order == release

            threadfence_system()
        else
            throw(AtomicOrderUnsupported(order))
        end
    end
end

"""
    threadfence(::SyncScope=device_scope)

A memory fence that ensures that:
- All writes to all memory made by the calling thread before the call to `threadfence(scope)`
  are observed by all threads in the scope of the calling thread as occurring before all writes
  to all memory made by the calling thread after the call to `threadfence(scope)`
- All reads from all memory made by the calling thread before the call to `threadfence(scope)`
  are ordered before all reads from all memory made by the calling thread after the call to `threadfence(scope)`.

SyncScope can be one of `block_scope`, `device_scope`, or `system_scope`.
  - `block_scope` orders reads and write on the *same* block.
  - `device_scope` orders reads and write on the *same* device.
  - `system_scope` orders reads and writes across all threads in the device,
    host threads, and all threads in peer devices.

See [`atomic_thread_fence`](@ref) for a variant that takes atomic orderings.

!!! note
  Note that for this ordering guarantee to be true, the observing threads must truly observe the
  memory and not cached versions of it; this is requires the use of atomic loads and stores.

"""
function threadfence end

"""
    atomic_thread_fence(order::Atomicx.Ordering, ::SyncScope=device)
"""
function atomic_thread_fence end