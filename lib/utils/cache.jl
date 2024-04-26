# a cache for library handles

# TODO:
# - keep track of the (estimated?) size of cache contents
# - clean the caches when memory is needed. this will require registering the destructor
#   upfront, so that it can set the environment (e.g. switch to the appropriate context).
#   alternatively, register the `unsafe_free!`` methods with the pool instead of the cache.

export HandleCache

struct HandleCache{K,V}
    ctor
    dtor

    active_handles::Set{Pair{K,V}}      # for debugging, and to prevent handle finalization
    idle_handles::Dict{K,Vector{V}}
    lock::ReentrantLock

    max_entries::Int

    function HandleCache{K,V}(ctor, dtor; max_entries::Int=32) where {K,V}
        return new{K,V}(ctor, dtor, Set{Pair{K,V}}(), Dict{K,Vector{V}}(), ReentrantLock(), max_entries)
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
