# C. Cooperative Groups

export this_grid, sync_grid

"""
    this_grid()

Returns a `grid_handle` of the grid group this thread belongs to. Only available if a
cooperative kernel is launched.
"""
this_grid() = cudaCGGetIntrinsicHandle(cudaCGScopeGrid)

"""
    sync_grid(grid_handle::Culonglong)

Waits until all threads in all blocks in the grid `grid_handle` have reached this point and
all global memory accesses made by these threads prior to `sync_grid()` are visible to all
threads in the grid. A 32-bit integer `cudaError_t` is returned.
"""
sync_grid(handle) = cudaCGSynchronize(handle, 0)
