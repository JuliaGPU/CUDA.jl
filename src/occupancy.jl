export active_blocks, occupancy, launch_configuration

"""
    active_blocks(fun::CuFunction, threads; shmem=0)

Calculate the maximum number of active blocks per multiprocessor when running `threads`
threads of a kernel `fun` requiring `shmem` bytes of dynamic shared memory.
"""
function active_blocks(fun::CuFunction, threads::Integer; shmem::Integer=0)
    blocks_ref = Ref{Cint}()
    @apicall(:cuOccupancyMaxActiveBlocksPerMultiprocessor,
             (Ptr{Cint}, CuFunction_t, Cint, Csize_t),
             blocks_ref, fun, threads, shmem)
    return blocks_ref[]
end

"""
    occupancy(fun::CuFunction, threads; shmem=0)

Calculate the theoretical occupancy of launching `threads` threads of a kernel `fun`
requiring `shmem` bytes of dynamic shared memory.

"""
function occupancy(fun::CuFunction, threads::Integer; shmem::Integer=0)
    # https://devblogs.nvidia.com/cuda-pro-tip-occupancy-api-simplifies-launch-configuration/
    blocks = active_blocks(fun, threads; shmem=shmem)

    mod = fun.mod
    ctx = mod.ctx
    dev = device(ctx)

    threads_per_sm = attribute(dev, MAX_THREADS_PER_MULTIPROCESSOR)
    warp_size = attribute(dev, WARP_SIZE)

    return (blocks * threads ÷ warp_size) / (threads_per_sm ÷ warp_size)
end

"""
    launch_configuration(fun::CuFunction; shmem=0, max_threads=0)

Calculate a suggested launch configuration for kernel `fun` requiring `shmem` bytes of
dynamic shared memory. Returns a tuple with a suggested amount of threads, and the minimal
amount of blocks to reach maximal occupancy. Optionally, the maximum amount of threads can
be constrained using `max_threads`.

In the case of a variable amount of shared memory, pass a callable object for `shmem`
instead, taking a single integer representing the block size and returning the amount of
dynamic shared memory for that configuration.
"""
function launch_configuration(fun::CuFunction; shmem=0, max_threads::Integer=0)
    blocks_ref = Ref{Cint}()
    threads_ref = Ref{Cint}()
    if isa(shmem, Integer)
        @apicall(:cuOccupancyMaxPotentialBlockSize,
                (Ptr{Cint}, Ptr{Cint}, CuFunction_t, Ptr{Cvoid}, Csize_t, Cint),
                blocks_ref, threads_ref, fun, C_NULL, shmem, max_threads)
    else
        shmem_cint = threads -> Cint(shmem(threads))
        cb = @cfunction($shmem_cint, Cint, (Cint,))
        @apicall(:cuOccupancyMaxPotentialBlockSize,
                (Ptr{Cint}, Ptr{Cint}, CuFunction_t, Ptr{Cvoid}, Csize_t, Cint),
                blocks_ref, threads_ref, fun, cb, 0, max_threads)
    end
    return (blocks=blocks_ref[], threads=threads_ref[])
end
