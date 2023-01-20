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

# remove a handle from the cache, or create a new one
function Base.pop!(f::Function, cache::HandleCache{K,V}, key) where {K,V}
    function check_cache(f::Function=()->nothing)
        lock(cache.lock) do
            handle = if !haskey(cache.idle_handles, key) || isempty(cache.idle_handles[key])
                f()
            else
                pop!(cache.idle_handles[key])
            end

            if handle !== nothing
                push!(cache.active_handles, key=>handle)
            end

            return handle
        end
    end

    handle = check_cache()

    if handle === nothing
        # if we didn't find anything, perform a quick GC collection to free up old handles.
        GC.gc(false)

        handle = check_cache(f)
    end

    return handle::V
end

# put a handle in the cache, or destroy it if it doesn't fit
function Base.push!(f::Function, cache::HandleCache{K,V}, key::K, handle::V) where {K,V}
    lock(cache.lock) do
        delete!(cache.active_handles, key=>handle)

        if haskey(cache.idle_handles, key)
            if length(cache.idle_handles[key]) > cache.max_entries
                f()
            else
                push!(cache.idle_handles[key], handle)
            end
        else
            cache.idle_handles[key] = [handle]
        end
    end
end

# shorthand version to put a handle back without having to remember the key
function Base.push!(f::Function, cache::HandleCache{K,V}, handle::V) where {K,V}
    lock(cache.lock) do
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
        push!(f, cache, key, handle)
    end
end
