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
# `Reclaimable` unifies both. Instances are registered via
# `register_reclaimable!` — typically from a library's `__init__`, since
# mutations to this registry performed during the precompilation of a
# downstream package don't carry over to module load (see
# `register_reclaimable!` for details).

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

Register `r` so that `reclaim` invokes its `drop!` and `purge!` methods.
Idempotent: re-registering the same instance is a no-op. Returns `r`.

Call this from a package's `__init__`: top-level constructor calls are
captured by Julia's precompilation cache, but mutations they perform on
dependencies (like pushing into CUDACore's registry) do not persist to
module load time.
"""
function register_reclaimable!(r::Reclaimable)
    @lock reclaimables_lock begin
        r in reclaimables || push!(reclaimables, r)
    end
    return r
end

# invoke f(r) per Reclaimable with per-hook error isolation, modelled
# after Base's `atexit_hooks`: one bad hook mustn't prevent the others.
function foreach_reclaimable(f)
    @lock reclaimables_lock begin
        for r in reclaimables
            try
                f(r)
            catch ex
                @error "reclaim callback failed" type=typeof(r) exception=(ex, catch_backtrace())
            end
        end
    end
end


## task-local state helper

"""
    TaskLocalCache{K,V}(key::Symbol)

Declarative marker for a library's per-task cache stored under `key` in
`task_local_storage()`. Must be `register_reclaimable!`'d from the owning
package's `__init__` so that `RECLAIM_DROP_STATE` clears the current
task's entry, letting the stored values (typically mutable handle wrappers)
be garbage-collected. Their finalizers then return the underlying
resources to a `HandleCache`.

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
    get!(() -> Dict{K,V}(), task_local_storage(), s.key)::Dict{K,V}
end

function drop!(s::TaskLocalCache)
    delete!(task_local_storage(), s.key)
    return
end
