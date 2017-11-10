# Synchronization (B.6)

export sync_threads, sync_warp

"""
    sync_threads()

Waits until all threads in the thread block have reached this point and all global and
shared memory accesses made by these threads prior to `sync_threads()` are visible to all
threads in the block.
"""
@inline sync_threads() = ccall("llvm.nvvm.barrier0", llvmcall, Void, ())

"""
    sync_warp(mask::Integer=0xffffffff)

Waits threads in the warp, selected by means of the bitmask `mask`, have reached this point
and all global and shared memory accesses made by these threads prior to `sync_warp()` are
visible to those threads in the warp. The default value for `mask` selects all threads in
the warp.
"""
sync_warp

if cuda_driver_version >= v"9.0" && v"6.0" in ptx_support
    @inline function sync_warp(mask::Integer=0xffffffff)
        return Base.llvmcall(
            """call void asm sideeffect "bar.warp.sync \$0;", "r"(i32 %0)
               ret void""",
            Void, Tuple{UInt32}, convert(UInt32, mask))
    end
else
    @inline sync_warp(mask::Integer=0xffffffff) = nothing
end
