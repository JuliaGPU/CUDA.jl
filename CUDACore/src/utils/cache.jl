# a cache for library handles

export HandleCache

# a handle held in task-local state is released at most once: when the garbage collector
# collects the wrapper holding it, or when another task needs a handle after the owning
# task has finished. whichever claims it first releases it.
mutable struct HandleClaim
    Base.@atomic claimed::Bool
end
claim!(c::HandleClaim) = (Base.@atomicreplace c.claimed false => true).success

struct HandleOwner
    task::WeakRef
    wrapper::WeakRef    # weak, so that the wrapper can still be collected
    claim::HandleClaim
    release::Any
    ctx::Any            # a `CuContext` or `nothing`, see `ReleaseAction`
end

struct HandleCache{K,V} <: Reclaimable
    ctor
    dtor

    active_handles::Set{Pair{K,V}}
    idle_handles::Dict{K,Vector{V}}
    # active handles held in task-local state (see `handle_finalizer`)
    owners::Dict{Pair{K,V},HandleOwner}
    lock::ReentrantLock

    # how many handles may be active before collecting garbage to try and free some
    gc_threshold::Int

    function HandleCache{K,V}(ctor, dtor; gc_threshold::Int=32) where {K,V}
        return new{K,V}(ctor, dtor, Set{Pair{K,V}}(), Dict{K,Vector{V}}(),
                        Dict{Pair{K,V},HandleOwner}(), ReentrantLock(),
                        gc_threshold)
    end
end

# destroying idle handles is the `purge!` step of reclaim; individual caches
# must be registered in the owning library's __init__ (see reclaim.jl).
purge!(cache::HandleCache) = Base.invokelatest(empty!, cache)

# take an idle handle for `key`, if any, also returning the number of active handles.
# (not inlined, as Julia 1.10 miscompiles the loop in `pop!` that calls this, crashing)
@noinline function take_idle!(cache::HandleCache{K,V}, key::K) where {K,V}
    @lock cache.lock begin
        handles = get(cache.idle_handles, key, nothing)
        handle = handles === nothing || isempty(handles) ? nothing : pop!(handles)
        handle, length(cache.active_handles)
    end
end

# remove a handle from the cache, or create a new one
function Base.pop!(cache::HandleCache{K,V}, key::K) where {K,V}
    drain_retired(ALLOC_DRAIN_LIMIT)
    # check the cache
    handle, num_active_handles = take_idle!(cache, key)

    # if we didn't find anything, take one back from a task that has finished. the garbage
    # collector is in no hurry to collect finished tasks, so code that spawns many tasks
    # (e.g. a threaded `map`) would otherwise keep creating handles. another task may take
    # the handle before we do, so keep trying.
    while handle === nothing && reclaim_finished!(cache, key)
        handle, num_active_handles = take_idle!(cache, key)
    end

    # if we didn't find anything, but lots of handles are active, try to free some
    if handle === nothing && num_active_handles > cache.gc_threshold
        GC.gc(false)
        drain_retired(ALLOC_DRAIN_LIMIT)
        handle, _ = take_idle!(cache, key)
    end

    # if we still didn't find anything, create a new handle
    if handle === nothing
        maybe_collect()
        handle = cache.ctor(key)
    end

    # add the handle to the active set
    @lock cache.lock begin
        push!(cache.active_handles, key=>handle)
    end

    return handle::V
end

# put a handle back in the cache. this is typically done from a finalizer, which can't take
# the cache's lock, so it is deferred. handles are only destroyed when memory is reclaimed,
# after releasing held resources that may depend on them (e.g. sparse analysis infos).
function Base.push!(cache::HandleCache{K,V}, key::K, handle::V) where {K,V}
    defer_release(args -> return_handle!(args...), (cache, key, handle))
    return
end

function return_handle!(cache::HandleCache{K,V}, key::K, handle::V) where {K,V}
    @lock cache.lock begin
        delete!(cache.owners, key=>handle)
        delete!(cache.active_handles, key=>handle)
        push!(get!(Vector{V}, cache.idle_handles, key), handle)
    end
    return
end

"""
    handle_finalizer(release, wrapper, cache::HandleCache, key, handle; ctx=context()) -> wrapper

Register `release(wrapper)` to put `handle`, obtained with `pop!(cache, key)`, back in the
cache once `wrapper` is collected, like [`resource_finalizer`](@ref) with `blocking=false`.
`wrapper` holds the handle in the current task's local state, so once the task has finished,
`pop!` may also call `release` to reuse the handle, instead of waiting for the garbage
collector to collect the task. Either way, `release` is called at most once.

`release` must not capture `wrapper` or wait for the GPU, and should put the handle back
with `push!(cache, key, handle)` after releasing anything else. Neither the handle nor what
is released along with it may be used after the task has finished (e.g., by returning it
from the task).
"""
function handle_finalizer(release, wrapper, cache::HandleCache{K,V}, key::K, handle::V;
                          ctx=context()) where {K,V}
    claim = HandleClaim(false)
    resource_finalizer(wrapper; blocking=false, ctx) do wrapper
        claim!(claim) && release(wrapper)
    end
    @lock cache.lock begin
        cache.owners[key=>handle] =
            HandleOwner(WeakRef(current_task()), WeakRef(wrapper), claim, release, ctx)
    end
    return wrapper
end

# release a handle for `key` whose task has finished, returning whether one was found
function reclaim_finished!(cache::HandleCache{K,V}, key::K) where {K,V}
    # (releasing is deferred while capturing)
    active_captures[] > 0 && return false
    owner, wrapper = @lock cache.lock begin
        found = nothing, nothing
        stale = Pair{K,V}[]
        for (entry, owner) in cache.owners
            wrapper = owner.wrapper.value
            if wrapper === nothing
                # collected, so its finalizer takes care of it
                push!(stale, entry)
                continue
            end
            entry.first == key || continue
            task = owner.task.value
            # (the wrappers of a collected task are released by the garbage collector)
            task !== nothing && istaskdone(task) || continue
            push!(stale, entry)
            found = owner, wrapper
            break
        end
        foreach(entry -> delete!(cache.owners, entry), stale)
        found
    end
    owner === nothing && return false
    # (unless the garbage collector claimed it first)
    if claim!(owner.claim)
        discard(ReleaseAction(owner.release, wrapper, owner.ctx, false))
    end
    return true
end

# shorthand version to put a handle back without having to remember the key
function Base.push!(cache::HandleCache{K,V}, handle::V) where {K,V}
    defer_release(args -> return_handle!(args...), (cache, handle))
    return
end

function return_handle!(cache::HandleCache{K,V}, handle::V) where {K,V}
    key = @lock cache.lock begin
        key = nothing
        for entry in cache.active_handles
            if entry[2] == handle
                key = entry[1]
                break
            end
        end
        if key === nothing
            error("Attempt to cache handle $handle that was not created by the handle cache")
        end
        key
    end

    return_handle!(cache, key, handle)
end

# empty the cache
# XXX: often we only need to empty the handles for a single context, however, we don't
#      know for sure that the key is a context (see e.g. cuFFT), so we wipe everything
function Base.empty!(cache::HandleCache{K,V}) where {K,V}
    handles = @lock cache.lock begin
        all_handles = Pair{K,V}[]
        for (key, handles) in cache.idle_handles, handle in handles
            push!(all_handles, key=>handle)
        end
        empty!(cache.idle_handles)
        all_handles
    end

    for (key,handle) in handles
        attempt_release(ReleaseAction(key=>handle, nothing, true) do (key, handle)
            cache.dtor(key, handle)
        end)
    end
end

# Counts rather than bytes: libraries do not expose all internal handle allocations.
function handle_cache_counts(cache::HandleCache)
    @lock cache.lock (active=length(cache.active_handles),
                     idle=sum(length, values(cache.idle_handles); init=0))
end
