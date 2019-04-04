# C. Cooperative Groups

export this_grid, sync_grid

"""
    this_grid()

Returns a `grid_handle` of the grid group this thread belongs to. Only available if a cooperative
kernel is launched.
"""
this_grid

if VERSION >= v"1.2.0-DEV.512"
    @inline this_grid() = ccall("extern cudaCGGetIntrinsicHandle", llvmcall, Culonglong, (Cuint,), UInt32(1))
else
    @eval @inline this_grid() = Base.llvmcall(
        ( "declare i64 @cudaCGGetIntrinsicHandle(i32)",
         $"%rv = call i64 @cudaCGGetIntrinsicHandle(i32 1)
           ret i64 %rv"), Culonglong,
        Tuple{})
end

"""
    sync_grid(grid_handle::Culonglong)

Waits until all threads in all blocks in the grid `grid_handle` have reached this point and all
global memory accesses made by these threads prior to `sync_grid()` are visible to all threads
in the grid. A 32-bit integer `cudaError_t` is returned.
"""
sync_grid

if VERSION >= v"1.2.0-DEV.512"
    @inline sync_grid(grid_handle::Culonglong) =
        ccall("extern cudaCGSynchronize", llvmcall, cudaError_t,
              (Culonglong, Cuint), grid_handle, UInt32(0))
else
    @eval @inline sync_grid(grid_handle::Culonglong) = Base.llvmcall(
        ( "declare i32 @cudaCGSynchronize(i64, i32)",
         $"%rv = call i32 @cudaCGSynchronize(i64 %0, i32 0)
           ret i32 %rv"), cudaError_t,
        Tuple{Culonglong}, grid_handle)
end
