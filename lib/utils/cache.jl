# a cache for library handles

export HandleCache

struct HandleCache{K,V}
    ctor
    dtor

    active_handles::Set{Pair{K,V}}
    idle_handles::Dict{K,Vector{V}}
    lock::Base.ThreadSynchronizer
    # XXX: we use a thread-safe spinlock because the handle cache is used from finalizers.
    #      once finalizers run on their own thread, use a regular ReentrantLock

    max_entries::Int

    function HandleCache{K,V}(ctor, dtor; max_entries::Int=32) where {K,V}
        obj = new{K,V}(ctor, dtor, Set{Pair{K,V}}(), Dict{K,Vector{V}}(),
                       Base.ThreadSynchronizer(), max_entries)

        # register a hook to wipe the current context's cache when under memory pressure
        push!(CUDA.reclaim_hooks, ()->empty!(obj))

        return obj
    end
end

# remove a handle from the cache, or create a new one
function Base.pop!(cache::HandleCache{K,V}, key::K) where {K,V}
    # check the cache
    handle = @lock cache.lock begin
        if !haskey(cache.idle_handles, key) || isempty(cache.idle_handles[key])
            nothing
        else
            pop!(cache.idle_handles[key])
        end
    end

    # if we didn't find anything, create a new handle.
    # we could (and used to) run `GC.gc(false)` here to free up old handles,
    # but that can be expensive when using lots of short-lived tasks.
    if handle === nothing
        CUDA.maybe_collect()
        handle = cache.ctor(key)
    end

    # add the handle to the active set
    @lock cache.lock begin
        push!(cache.active_handles, key=>handle)
    end

    return handle::V
end

# put a handle in the cache, or destroy it if it doesn't fit
function Base.push!(cache::HandleCache{K,V}, key::K, handle::V) where {K,V}
    saved = @lock cache.lock begin
        delete!(cache.active_handles, key=>handle)

        if haskey(cache.idle_handles, key)
            if length(cache.idle_handles[key]) > cache.max_entries
                false
            else
                push!(cache.idle_handles[key], handle)
                true
            end
        else
            cache.idle_handles[key] = [handle]
            true
        end
    end

    if !saved
        cache.dtor(key, handle)
    end
end

# shorthand version to put a handle back without having to remember the key
function Base.push!(cache::HandleCache{K,V}, handle::V) where {K,V}
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

    push!(cache, key, handle)
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
        cache.dtor(key, handle)
    end
end
