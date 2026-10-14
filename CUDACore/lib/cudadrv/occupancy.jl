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
function _shmem_cint_cb(x::Cint)
    Cint(something(_shmem_cb)(x))
end
_shmem_cb_lock = Threads.ReentrantLock()

"""
    launch_configuration(fun::CuFunction; shmem=0, max_threads=0)

Calculate a suggested launch configuration for kernel `fun` requiring `shmem` bytes of
dynamic shared memory. Returns a tuple with a suggested amount of threads, and the minimal
amount of blocks to reach maximal occupancy. Optionally, the maximum amount of threads can
be constrained using `max_threads`.

In the case of a variable amount of shared memory, pass a callable object for `shmem`
instead, taking a single integer representing the block size and returning the amount of
dynamic shared memory for that configuration.

Without dynamic shared memory, the result is cached in `fun`, and reused for every
`max_threads` that admits the unconstrained number of threads. Setting a function attribute
with `attributes` or the cache configuration with `cache_config!` invalidates
the cache, but changing the configuration through the driver API directly (e.g.
`cuFuncSetCacheConfig`) does not.
"""
function launch_configuration(fun::CuFunction; shmem::Union{Integer,Base.Callable}=0,
                              max_threads::Integer=0)
    max_threads = clamp(max_threads, Cint)
    if shmem isa Integer && shmem == 0
        config = cached_launch_configuration(fun)
        # the driver picks the largest block size with the best occupancy, so a limit that
        # admits the unconstrained choice doesn't change the result
        if max_threads == 0 || max_threads >= config.threads
            return (blocks=config.blocks, threads=config.threads)
        end
    end
    return query_launch_configuration(fun, shmem, max_threads)
end

# incremented whenever something that affects launch configurations changes, invalidating
# the cached ones. global rather than per function, because the context's cache configuration
# applies to all functions, and several `CuFunction` objects can share a handle.
const launch_config_generation = Threads.Atomic{Int}(0)

# to be called after making the change, so that a configuration cached under the new
# generation reflects it
invalidate_launch_configurations() = (Threads.atomic_add!(launch_config_generation, 1); nothing)

struct CachedLaunchConfiguration
    generation::Int
    blocks::Int
    threads::Int
end

function cached_launch_configuration(fun::CuFunction)
    # read the generation before querying, so that a change concurrent with the query makes
    # the result stale instead of getting cached as current
    generation = launch_config_generation[]
    config = Base.@atomic :acquire fun.launch_config
    if !(config isa CachedLaunchConfiguration && config.generation == generation)
        (; blocks, threads) = query_launch_configuration(fun, 0, Cint(0))
        config = CachedLaunchConfiguration(generation, blocks, threads)
        Base.@atomic :release fun.launch_config = config
    end
    return config::CachedLaunchConfiguration
end

function query_launch_configuration(fun::CuFunction, shmem::Union{Integer,Base.Callable},
                                    max_threads::Cint)
    blocks_ref = Ref{Cint}()
    threads_ref = Ref{Cint}()
    if isa(shmem, Integer)
        cuOccupancyMaxPotentialBlockSize(blocks_ref, threads_ref, fun, C_NULL, shmem, max_threads)
    elseif Sys.ARCH == :x86 || Sys.ARCH == :x86_64
        function shmem_cint(threads)
            Cint(shmem(threads))
        end
        cb = @cfunction($shmem_cint, Cint, (Cint,))
        cuOccupancyMaxPotentialBlockSize(blocks_ref, threads_ref, fun, cb, 0, max_threads)
    else
        lock(_shmem_cb_lock) do
            global _shmem_cb
            _shmem_cb = shmem
            cb = @cfunction(_shmem_cint_cb, Cint, (Cint,))
            cuOccupancyMaxPotentialBlockSize(blocks_ref, threads_ref, fun, cb, 0, max_threads)
            _shmem_cb = nothing
        end
    end
    return (blocks=Int(blocks_ref[]), threads=Int(threads_ref[]))
end
