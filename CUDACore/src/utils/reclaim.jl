# reclaim registry
#
# Libraries (cuBLAS, cuDNN, …) hold GPU resources that the memory subsystem
# needs to release under pressure. Two kinds of resources exist:
#
#  - live state referenced by a running task (typically via
#    `task_local_storage`), e.g. fat handles wrapping library handles plus
#    workspace buffers. Released via `drop!` — the library clears its TLS
#    references so the wrappers become GC-eligible; their object-bound
#    finalizers then return the raw handles to a `HandleCache`;
#
#  - idle resources cached for reuse, e.g. the `HandleCache` of previously-
#    returned library handles. Released via `purge!` — the cache is
#    emptied and each entry's destructor runs.
#
# `Reclaimable` unifies both. Register instances via `register_reclaimable!`.

abstract type Reclaimable end

"""
    CUDACore.drop!(r::Reclaimable)

Release references to live state so the associated GC-managed wrappers
become collectible (and their finalizers can run). Default: do nothing.
"""
drop!(::Reclaimable) = nothing

"""
    CUDACore.purge!(r::Reclaimable)

Destroy idle cached resources owned by `r`. Default: do nothing.
"""
purge!(::Reclaimable) = nothing

const reclaimables = Reclaimable[]
const reclaimables_lock = ReentrantLock()

"""
    CUDACore.register_reclaimable!(r::Reclaimable)

Register `r` so `reclaim` invokes its `drop!` and `purge!` methods.
Idempotent; returns `r`. Call from the owning package's `__init__`
(registry mutations performed during precompile don't survive to load).
"""
function register_reclaimable!(r::Reclaimable)
    @lock reclaimables_lock begin
        r in reclaimables || push!(reclaimables, r)
    end
    return r
end

# Snapshot the registry under the lock, then run callbacks unlocked
# (Base.atexit-style): callbacks may take a while and may transitively
# trigger reclaim, so we don't want to hold the lock across them.
function foreach_reclaimable(f)
    snapshot = @lock reclaimables_lock copy(reclaimables)
    for r in snapshot
        try
            f(r)
        catch ex
            @error "reclaim callback failed" type=typeof(r) exception=(ex, catch_backtrace())
        end
    end
end


## task-local state helper

"""
    TaskLocalCache{K,V}(key::Symbol)

Declarative marker for a library's per-task cache stored under `key` in
`task_local_storage()`. Must be `register_reclaimable!`'d from the owning
package's `__init__` so that `RECLAIM_DROP` clears the current task's
entry, letting the stored values (typically mutable handle wrappers) be
garbage-collected. Their finalizers then return the underlying resources
to a `HandleCache`.

Usage:

    const state_cache = CUDACore.TaskLocalCache{CuContext, LibraryState}(:CUBLAS)

    function __init__()
        ...
        CUDACore.register_reclaimable!(state_cache)
    end

    function handle()
        states = CUDACore.task_dict(state_cache)
        state = get!(() -> new_state(...), states, key)
        ...
    end

Only the current task's storage is touched: Julia's `IdDict`-backed TLS
isn't safe for concurrent mutation across threads, so cross-task drops are
intentionally deferred to task GC (values become unreachable when the
task is collected).
"""
struct TaskLocalCache{K,V} <: Reclaimable
    key::Symbol
    TaskLocalCache{K,V}(key::Symbol) where {K,V} = new{K,V}(key)
end

@inline function task_dict(s::TaskLocalCache{K,V}) where {K,V}
    # spelled out instead of get!(()->Dict, ...) to avoid the closure alloc
    tls = task_local_storage()
    d = get(tls, s.key, nothing)
    if d === nothing
        d = Dict{K,V}()
        tls[s.key] = d
    end
    return d::Dict{K,V}
end

function drop!(s::TaskLocalCache)
    delete!(task_local_storage(), s.key)
    return
end
