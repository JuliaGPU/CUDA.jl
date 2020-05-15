export active_blocks, occupancy, launch_configuration

"""
    active_blocks(fun::CuFunction, threads; shmem=0)

Calculate the maximum number of active blocks per multiprocessor when running `threads`
threads of a kernel `fun` requiring `shmem` bytes of dynamic shared memory.
"""
function active_blocks(fun::CuFunction, threads::Integer; shmem::Integer=0)
    blocks_ref = Ref{Cint}()
    cuOccupancyMaxActiveBlocksPerMultiprocessor(blocks_ref, fun, threads, shmem)
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

    threads_per_sm = attribute(dev, DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR)
    warp_size = attribute(dev, DEVICE_ATTRIBUTE_WARP_SIZE)

    return (blocks * threads ÷ warp_size) / (threads_per_sm ÷ warp_size)
end

# HACK: callback function for `launch_configuration` on platforms without support for
#       trampolines as used by `@cfunction` (JuliaLang/julia#27174, JuliaLang/julia#32154)
_shmem_cb = nothing
_shmem_cint_cb(x::Cint) = Cint(something(_shmem_cb)(x))

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
function launch_configuration(fun::CuFunction; shmem::Union{Integer,Base.Callable}=0,
                              max_threads::Integer=0)
    blocks_ref = Ref{Cint}()
    threads_ref = Ref{Cint}()
    if isa(shmem, Integer)
        cuOccupancyMaxPotentialBlockSize(blocks_ref, threads_ref, fun, C_NULL, shmem, max_threads)
    elseif Sys.ARCH == :x86 || Sys.ARCH == :x86_64
        shmem_cint = threads -> Cint(shmem(threads))
        cb = @cfunction($shmem_cint, Cint, (Cint,))
        cuOccupancyMaxPotentialBlockSize(blocks_ref, threads_ref, fun, cb, 0, max_threads)
    else
        global _shmem_cb
        _shmem_cb = shmem
        cb = @cfunction(_shmem_cint_cb, Cint, (Cint,))
        cuOccupancyMaxPotentialBlockSize(blocks_ref, threads_ref, fun, cb, 0, max_threads)
        _shmem_cb = nothing
    end
    return (blocks=blocks_ref[], threads=threads_ref[])
end
