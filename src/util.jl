## version range

struct VersionRange
    lower::VersionNumber
    upper::VersionNumber
end

Base.in(v::VersionNumber, r::VersionRange) = (v >= r.lower && v <= r.upper)

Base.colon(a::VersionNumber, b::VersionNumber) = VersionRange(a, b)

Base.intersect(v::VersionNumber, r::VersionRange) =
    v < r.lower ? (r.lower:v) :
    v > r.upper ? (v:r.upper) : (v:v)


## logging

const DEBUG = parse(Bool, get(ENV, "DEBUG", "false"))

if VERSION >= v"0.7-"
    using Logging
    DEBUG && global_logger(SimpleLogger(global_logger().stream, Logging.Debug))
else
    export @debug
    @inline function debug(io::IO, msg...; prefix="DEBUG: ", line=true)
        @static if DEBUG
            print_with_color(:green, io, prefix, chomp(string(msg...)), line ? "\n" : "")
        end
    end
    @inline debug(msg...; kwargs...) = debug(STDERR, msg...; kwargs...)
    macro debug(args...)
        DEBUG && return Expr(:call, :debug, map(esc, args)...)
    end
end
