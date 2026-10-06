## reclaim escalation
#
# `Reclaimable`/`register_reclaimable!`/`TaskLocalCache`/`drop!`/`purge!`
# are defined in utils/reclaim.jl. Here we add the ladder that drives them
# along with the allocator-specific sync/trim steps.
#
# Registered `drop!`/`purge!` callbacks must not switch the active device:
# the device is captured once per `reclaim` / `retry_reclaim` call.

"""
    ReclaimLevel

Escalation levels shared by `reclaim(level)` and `retry_reclaim`, from
cheapest to most aggressive:

| Level           | Action                                              |
| :---            | :---                                                |
| `RECLAIM_PURGE` | empty `HandleCache`s (no GC, no sync)               |
| `RECLAIM_SYNC`  | synchronize the device (lets async deallocs finish) |
| `RECLAIM_GC`    | run a full Julia GC, then sync + purge + trim       |
| `RECLAIM_DROP`  | also drop task-local library state, then GC + …     |

`RECLAIM_DROP` clears the calling task's library state — see
[`register_reclaimable!`](@ref). It assumes user-held descriptors/plans
are tied to a context, not to a specific library-handle instance (true
for all libraries CUDA.jl wraps). Live handles the user holds a
reference to are unaffected: the wrapper keeps the raw handle alive.

Steps that don't apply to the current allocator (e.g. trim on a
non-stream-ordered device) are silently skipped.
"""
@enum ReclaimLevel::Int begin
    RECLAIM_PURGE = 0
    RECLAIM_SYNC  = 1
    RECLAIM_GC    = 2
    RECLAIM_DROP  = 3
end


"""
    retry_reclaim(retry_if) do
        # code that may fail due to insufficient GPU memory
    end

Run a block of code repeatedly while `retry_if(ret)` holds for its return
value, escalating one `ReclaimLevel` between attempts. Returns the final
(or most recent) return value of the block.

This is intended for CUDA APIs that allocate outside the pool and report
failure via a status code. It's like `Base.retry`, but works on return
values instead of exceptions for performance reasons.
"""
@inline function retry_reclaim(f, retry_if)
    ret = f()
    retry_if(ret) || return ret
    return retry_reclaim_slow(f, retry_if, ret)
end

@noinline function retry_reclaim_slow(f, retry_if, ret)
    dev = active_state().device
    so = stream_ordered(dev)
    for level in instances(ReclaimLevel)
        reclaim_step(level, dev, so)
        ret = f()
        retry_if(ret) || return ret
    end
    return ret
end

# Each level is a complete reclaim at that aggressiveness — `reclaim(level)`
# just runs the matching step. `retry_reclaim` walks the levels in order to
# bisect on alloc failure. GC.gc(true) drains pending finalizers before
# returning, so the post-GC purge sees caches populated by wrapper finalizers.
function reclaim_step(level::ReclaimLevel, dev::CuDevice, stream_ordered::Bool)
    if level == RECLAIM_PURGE
        foreach_reclaimable(purge!)
    elseif level == RECLAIM_SYNC
        stream_ordered && device_synchronize()
    elseif level == RECLAIM_GC
        GC.gc(true)
        stream_ordered && device_synchronize()
        foreach_reclaimable(purge!)
        stream_ordered && trim(pool_create(dev))
    elseif level == RECLAIM_DROP
        foreach_reclaimable(drop!)
        GC.gc(true)
        stream_ordered && device_synchronize()
        foreach_reclaimable(purge!)
        stream_ordered && trim(pool_create(dev))
    end
    return
end


"""
    reclaim([level::ReclaimLevel = RECLAIM_DROP])

Free GPU memory at the given [`ReclaimLevel`](@ref). The default drops
task-local library state, runs a full GC so handle wrappers finalize and
return their raw handles to caches, then destroys those caches and trims
the pool. Returns `nothing`.
"""
function reclaim(level::ReclaimLevel = RECLAIM_DROP)
    dev = active_state().device
    reclaim_step(level, dev, stream_ordered(dev))
    return
end
