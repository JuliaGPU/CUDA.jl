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

Identical to `__syncthreads()` with the additional feature that it evaluates predicate
for all threads of the block and returns the number of threads for which `predicate` evaluates
to non-zero.

    sync_threads_count(predicate::Bool)

Identical to `__syncthreads()` with the additional feature that it evaluates predicate
for all threads of the block and returns the number of threads for which `predicate` evaluates
to `true`.
"""
@inline sync_threads_count(predicate::Int32) = ccall("llvm.nvvm.barrier0.popc", llvmcall, Int32, (Int32,), predicate)
@inline sync_threads_count(predicate::Bool) = sync_threads_count(Int32(predicate))

"""
    sync_threads_and(predicate::Int32)

Identical to `__syncthreads()` with the additional feature that it evaluates predicate
for all threads of the block and returns non-zero if and only if `predicate` evaluates to
non-zero for all of them.

    sync_threads_and(predicate::Bool)

Identical to `__syncthreads()` with the additional feature that it evaluates predicate
for all threads of the block and returns `true` if and only if `predicate` evaluates to
`true` for all of them.
"""
@inline sync_threads_and(predicate::Int32) = ccall("llvm.nvvm.barrier0.and", llvmcall, Int32, (Int32,), predicate)
@inline sync_threads_and(predicate::Bool) = ifelse(sync_threads_and(Int32(predicate)) !== Int32(0), true, false)

"""
    sync_threads_or(predicate::Int32)

Identical to `__syncthreads()` with the additional feature that it evaluates predicate
for all threads of the block and returns non-zero if and only if `predicate` evaluates to
non-zero for any of them.

    sync_threads_or(predicate::Int32)

Identical to `__syncthreads()` with the additional feature that it evaluates predicate
for all threads of the block and returns `true` if and only if `predicate` evaluates to
`true` for any of them.
"""
@inline sync_threads_or(predicate::Int32) = ccall("llvm.nvvm.barrier0.or", llvmcall, Int32, (Int32,), predicate)
@inline sync_threads_or(predicate::Bool) = ifelse(sync_threads_or(Int32(predicate)) !== Int32(0), true, false)

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

"""
    threadfence_block()

A memory fence that ensures that:
- All writes to all memory made by the calling thread before the call to `threadfence_block()`
  are observed by all threads in the block of the calling thread as occurring before all writes
  to all memory made by the calling thread after the call to `threadfence_block()`
- All reads from all memory made by the calling thread before the call to `threadfence_block()`
  are ordered before all reads from all memory made by the calling thread after the call to `threadfence_block()`.
"""
@inline threadfence_block() = ccall("llvm.nvvm.membar.cta", llvmcall, Cvoid, ())

"""
    threadfence()

A memory fence that acts as [`threadfence_block`](@ref) for all threads in the block of the
calling thread and also ensures that no writes to all memory made by the calling thread after
the call to `threadfence()` are observed by any thread in the device as occurring before any
write to all memory made by the calling thread before the call to `threadfence()`.

Note that for this ordering guarantee to be true, the observing threads must truly observe the
memory and not cached versions of it; this is requires the use of volatile loads and stores,
which is not available from Julia right now.
"""
@inline threadfence() = ccall("llvm.nvvm.membar.gl", llvmcall, Cvoid, ())

"""
    threadfence_system()

A memory fence that acts as [`threadfence_block`](@ref) for all threads in the block of the
calling thread and also ensures that all writes to all memory made by the calling thread
before the call to `threadfence_system()` are observed by all threads in the device,
host threads, and all threads in peer devices as occurring before all writes to all
memory made by the calling thread after the call to `threadfence_system()`.
"""
@inline threadfence_system() = ccall("llvm.nvvm.membar.sys", llvmcall, Cvoid, ())
