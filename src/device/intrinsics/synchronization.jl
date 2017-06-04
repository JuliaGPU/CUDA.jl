# Synchronization (B.6)

export sync_threads

"""
    sync_threads()

Waits until all threads in the thread block have reached this point and all global and
shared memory accesses made by these threads prior to `sync_threads()` are visible to all
threads in the block.
"""
@inline sync_threads() = ccall("llvm.nvvm.barrier0", llvmcall, Void, ())
