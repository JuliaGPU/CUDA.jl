# a cache for library handles

export HandleCache

struct HandleCache{K,V} <: Reclaimable
    ctor
    dtor

    active_handles::Set{Pair{K,V}}
    idle_handles::Dict{K,Vector{V}}
    lock::ReentrantLock

    # how many handles may be active before collecting garbage to try and free some
    gc_threshold::Int

    function HandleCache{K,V}(ctor, dtor; gc_threshold::Int=32) where {K,V}
        return new{K,V}(ctor, dtor, Set{Pair{K,V}}(), Dict{K,Vector{V}}(),
                        ReentrantLock(), gc_threshold)
    end
end

# destroying idle handles is the `purge!` step of reclaim; individual caches
# must be registered in the owning library's __init__ (see reclaim.jl).
purge!(cache::HandleCache) = Base.invokelatest(empty!, cache)

# remove a handle from the cache, or create a new one
function Base.pop!(cache::HandleCache{K,V}, key::K) where {K,V}
    drain_retired(ALLOC_DRAIN_LIMIT)
    # check the cache
    handle, num_active_handles = @lock cache.lock begin
        if haskey(cache.idle_handles, key) && !isempty(cache.idle_handles[key])
            pop!(cache.idle_handles[key]), length(cache.active_handles)
        else
            nothing, length(cache.active_handles)
        end
    end

    # if we didn't find anything, but lots of handles are active, try to free some
    if handle === nothing && num_active_handles > cache.gc_threshold
        GC.gc(false)
        drain_retired(ALLOC_DRAIN_LIMIT)
        @lock cache.lock begin
            if haskey(cache.idle_handles, key) && !isempty(cache.idle_handles[key])
                handle = pop!(cache.idle_handles[key])
            end
        end
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
        delete!(cache.active_handles, key=>handle)
        push!(get!(Vector{V}, cache.idle_handles, key), handle)
    end
    return
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
