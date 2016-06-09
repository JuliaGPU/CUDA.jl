# Utilities

# Conditional logging (augmenting the default info/warn/error)
global TRACE = haskey(ENV, "TRACE")
@inline function trace(io::IO, msg...; prefix="TRACE: ", line=true)
    @static if TRACE
        Base.print_with_color(:cyan, io, prefix, chomp(string(msg...)))
        if line
            println(io)
        end
    end
end
@inline trace(msg...; prefix="TRACE: ", line=true) = trace(STDERR, msg..., prefix=prefix, line=line)

global DEBUG = TRACE || haskey(ENV, "DEBUG")
@inline function debug(io::IO, msg...; prefix="DEBUG: ")
    @static if DEBUG
        Base.println_with_color(:green, io, prefix, chomp(string(msg...)))
    end
end
@inline debug(msg...; prefix="DEBUG: ") = debug(STDERR, msg..., prefix=prefix)


function __init_util__()
    # TODO: assign TRACE and DEBUG at run-time, not using the pre-compiled code
    #       when the values are different?
    #       or make it work like CPU_CORES dose after Julia/#16219

    if TRACE
        trace("CUDAdrv.jl is running in trace mode, this will generate a lot of additional output")
    elseif DEBUG
        debug("CUDAdrv.jl is running in debug mode, this will generate additional output")
        debug("Run with TRACE=1 to enable even more output")
    end
end
