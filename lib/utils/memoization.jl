export @memoize

# simple, no-frills, thread-safe memoization helper
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

    # generate a cache and code to look-up values
    @gensym cache cache_lock cached_val val
    @eval __module__ begin
        const $cache_lock = ReentrantLock()
    end
    if isempty(args)
        @eval __module__ begin
            const $cache = Ref{Union{Nothing,$rettyp}}(nothing)
        end
        ex = quote
            global $cache
            $cached_val = Base.@lock $cache_lock $cache[]
            if $cached_val !== nothing
                $cached_val
            else
                $val = $code::$rettyp
                $cache[] = $val
                $val
            end
        end
    else
        if length(args) == 1
            @eval __module__ begin
                const $cache = Dict{$(argtyps[1]),$rettyp}()
            end
            key = :($(argvars[1]))
        else
            @eval __module__ begin
                const $cache = Dict{Tuple{$(argtyps...)},$rettyp}()
            end
            key = :(tuple($(argvars...)))
        end
        ex = quote
            $cached_val = Base.@lock $cache_lock get($cache, $key, nothing)
            if $cached_val !== nothing
                $cached_val
            else
                $val = $code::$rettyp
                $cache[$key] = $val
                $val
            end
        end
    end

    esc(ex)
end
