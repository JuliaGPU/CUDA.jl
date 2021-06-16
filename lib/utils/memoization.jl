export @memoize

"""
    @memoize [arg::T]... begin
        # expensive computation
    end::T

Low-level, no-frills memoization macro that stores values in a thread-local, typed Dict. The
types of the dictionary are derived from the syntactical type assertions.

The memoization is thread-safe, but instead of using locks (~30ns) an atomic status variable
is used. This improves performance to about 5ns per access. On the flip side, this means
that the functoin being memoized might get called multiple times, once per thread.

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
    # access to this cache is via an atomic variable so that we don't have to take a lock.
    @gensym global_cache_status global_cache

    # generate code to access memoized values
    # (assuming the global_cache can be indexed with the thread ID)
    if isempty(args)
        # if we don't have to key on anything, use the global cache directly
        global_cache_eltyp = :(Union{Nothing,$rettyp})
        global_init = :(nothing)
        local_ex = quote
            cached_value = @inbounds $(esc(global_cache))[Threads.threadid()]
            if cached_value !== nothing
                cached_value
            else
                new_value = $(esc(code))::$rettyp
                @inbounds $(esc(global_cache))[Threads.threadid()] = new_value
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
        local_ex = quote
            local_cache = @inbounds $(esc(global_cache))[Threads.threadid()]
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

    # generate code to handle the per-thread caches
    @eval __module__ begin
        # 0: initial state
        # 1: initializing
        # 2: initialized
        const $global_cache_status = Threads.Atomic{Int}(0)
    end
    @eval __module__ begin
        const $global_cache = $(global_cache_eltyp)[]
    end
    global_ex = quote
        while $(esc(global_cache_status))[] != 2
            status = Threads.atomic_cas!($(esc(global_cache_status)), 0, 1)
            if status == 0
                resize!($(esc(global_cache)), Threads.nthreads())
                for thread in 1:Threads.nthreads()
                    $(esc(global_cache))[thread] = $global_init
                end
                $(esc(global_cache_status))[] = 2
            else
                ccall(:jl_cpu_pause, Cvoid, ())
                # Temporary solution before we have gc transition support in codegen.
                ccall(:jl_gc_safepoint, Cvoid, ())
            end
        end
    end

    quote
        $global_ex
        $local_ex
    end
end
