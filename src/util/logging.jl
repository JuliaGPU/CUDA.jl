global TRACE = haskey(ENV, "TRACE")
"Display a trace message. Only results in actual printing if the TRACE environment variable
is set."
@inline function trace(io::IO, msg...; prefix="TRACE: ", line=true)
    @static if TRACE
        Base.print_with_color(:cyan, io, prefix, chomp(string(msg...)))
        if line
            println(io)
        end
    end
end
@inline trace(msg...; prefix="TRACE: ", line=true) = trace(STDERR, msg...; prefix=prefix, line=line)

global DEBUG = TRACE || haskey(ENV, "DEBUG")
"Display a debug message. Only results in actual printing if the TRACE or DEBUG environment
variable is set."
@inline function debug(io::IO, msg...; prefix="DEBUG: ", line=true)
    @static if DEBUG
        Base.print_with_color(:green, io, prefix, chomp(string(msg...)))
        if line
            println(io)
        end
    end
end
@inline debug(msg...; prefix="DEBUG: ", line=true) = debug(STDERR, msg...; prefix=prefix, line=line)

"Create an indented string from any value (instead of escaping endlines as \n)"
function repr_indented(ex; prefix=" "^7)
    io = IOBuffer()
    print(io, ex)
    str = takebuf_string(io)

    # Limit output
    if length(str) > 256
        if isa(ex, Array)
            T = eltype(ex)
            dims = join(size(ex), " by ")
            if zeros(ex) == ex
                str = "$T[$dims zeros]"
            else
                str = "$T[$dims elements]"
            end
        else
            if contains(strip(str), "\n")
                str = str[1:100] * "…\n\n[snip]\n\n…" * str[end-100:end]
            else
                str = str[1:100] * "…" * str[end-100:end]
            end
        end
    end

    lines = split(strip(str), '\n')
    if length(lines) > 1
        for i = 1:length(lines)
            lines[i] = prefix * lines[i]
        end

        lines[1] = "\"\n" * lines[1]
        lines[length(lines)] = lines[length(lines)] * "\""

        return join(lines, '\n')
    else
        return str
    end
end


function __init_logging__()
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
