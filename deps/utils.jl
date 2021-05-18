export getenv

# robustly get and parse an env var
function getenv(var, default::T) where T
    if haskey(ENV, var)
        result = tryparse(T, ENV[var])
        if result === nothing
            @warn "Could not parse $(var)=$(ENV[var]), using default value '$default'"
            default
        else
            result
        end
    else
        default
    end
end
