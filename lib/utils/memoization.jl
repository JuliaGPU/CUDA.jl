export @memoize

"""
    @memoize [arg::T]... begin
        # expensive computation
    end::T

Low-level, no-frills memoization macro that stores values in a thread-local, typed Dict. The
types of the dictionary are derived from the syntactical type assertions.

When there are no arguments to key the cache with, instead of a dictionary a simple array
with per-thread elements is used. This further improves performance to 2ns per access.
"""
macro memoize(ex...)
    code = ex[end]
    args = ex[1:end-1]

    # decode the code body
    @assert Meta.isexpr(code, :(::))
    rettyp = code.args[2]
    code = code.args[1]

    # decode the arguments
    argtyps = []
    argvars = []
    for arg in args
        @assert Meta.isexpr(arg, :(::))
        push!(argvars, arg.args[1])
        push!(argtyps, arg.args[2])
    end

    # the global cache is an array with one entry per thread. if we don't have to key on
    # anything, that entry will be the memoized new_value, or else a dictionary of values.
    @gensym global_cache

    # generate code to access memoized values
    # (assuming the global_cache can be indexed with the thread ID)
    if isempty(args)
        # if we don't have to key on anything, use the global cache directly
        global_cache_eltyp = :(Union{Nothing,$rettyp})
        ex = quote
            cache = get!($(esc(global_cache))) do
                [nothing for _ in 1:Threads.nthreads()]
            end
            cached_value = @inbounds cache[Threads.threadid()]
            if cached_value !== nothing
                cached_value
            else
                new_value = $(esc(code))::$rettyp
                @inbounds cache[Threads.threadid()] = new_value
                new_value
            end
        end
    else
        if length(args) == 1
            global_cache_eltyp = :(Dict{$(argtyps[1]),$rettyp})
            global_init = :(Dict{$(argtyps[1]),$rettyp}())
            key = :($(esc(argvars[1])))
        else
            global_cache_eltyp = :(Dict{Tuple{$(argtyps...)},$rettyp})
            global_init = :(Dict{Tuple{$(argtyps...)},$rettyp}())
            key = :(tuple($(map(esc, argvars)...)))
        end
        ex = quote
            cache = get!($(esc(global_cache))) do
                [$global_init for _ in 1:Threads.nthreads()]
            end
            local_cache = @inbounds cache[Threads.threadid()]
            cached_value = get(local_cache, $key, nothing)
            if cached_value !== nothing
                cached_value
            else
                new_value = $(esc(code))::$rettyp
                local_cache[$key] = new_value
                new_value
            end
        end
    end

    # define the per-thread cache
    @eval __module__ begin
        const $global_cache = LazyInitialized{Vector{$(global_cache_eltyp)}}()
    end

    quote
        $ex
    end
end
