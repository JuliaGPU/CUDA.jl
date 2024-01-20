# a cache for library handles

# TODO:
# - keep track of the (estimated?) size of cache contents
# - clean the caches when memory is needed. this will require registering the destructor
#   upfront, so that it can set the environment (e.g. switch to the appropriate context).
#   alternatively, register the `unsafe_free!`` methods with the pool instead of the cache.

export HandleCache

struct HandleCache{K,V}
    active_handles::Set{Pair{K,V}}      # for debugging, and to prevent handle finalization
    idle_handles::Dict{K,Vector{V}}
    lock::ReentrantLock

    max_entries::Int

    function HandleCache{K,V}(max_entries::Int=32) where {K,V}
        return new{K,V}(Set{Pair{K,V}}(), Dict{K,Vector{V}}(), ReentrantLock(), max_entries)
    end
end

# take and release a lock in a way that won't ever cause it to be locked during GC.
# this ensures that we don't have to worry about task switches during finalizers.
# note that the scope of this operation should be limited to the minimum possible.
macro safe_lock(l, ex)
    quote
        GC.enable(false)
        lock($(esc(l)))
        try
            $(esc(ex))
        finally
            unlock($(esc(l)))
            GC.enable(true)
        end
    end
end

# remove a handle from the cache, or create a new one
function Base.pop!(ctor::Function, cache::HandleCache{K,V}, key) where {K,V}
    # try to find an idle handle
    function check_cache()
        @safe_lock cache.lock begin
            if !haskey(cache.idle_handles, key) || isempty(cache.idle_handles[key])
                nothing
            else
                pop!(cache.idle_handles[key])
            end
        end
    end
    handle = check_cache()
    if handle === nothing
        GC.gc(false)
        handle = check_cache()
    end

    # if we didn't find anything, create a new handle
    if handle === nothing
        # `ctor` is an expensive function, and may trigger a GC collection,
        # so be sure to execute it outside of the lock.
        handle = ctor()::V
    end

    # mark the handle as active
    @safe_lock cache.lock begin
        push!(cache.active_handles, key=>handle)
    end

    return handle
end

# put a handle in the cache, or destroy it if it doesn't fit
function Base.push!(dtor::Function, cache::HandleCache{K,V}, key::K, handle::V) where {K,V}
    should_destroy = @safe_lock cache.lock begin
        in(key=>handle, cache.active_handles) || error("Cannot cache unknown handle $handle (key $key)")
        delete!(cache.active_handles, key=>handle)

        if haskey(cache.idle_handles, key)
            if length(cache.idle_handles[key]) > cache.max_entries
                true
            else
                push!(cache.idle_handles[key], handle)
                false
            end
        else
            cache.idle_handles[key] = [handle]
            false
        end
    end
    if should_destroy
        dtor()
    end
    return
end

# shorthand version to put a handle back without having to remember the key
function Base.push!(dtor::Function, cache::HandleCache{K,V}, handle::V) where {K,V}
    key = nothing
    @safe_lock cache.lock begin
        for entry in cache.active_handles
            if entry[2] == handle
                key = entry[1]
                break
            end
        end
    end
    if key === nothing
        error("Cannot cache unknown handle $handle")
    end
    push!(dtor, cache, key, handle)
    return
end
