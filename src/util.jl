export
    mkstemps

const dumpdir = Ref{String}()


# Conditional logging (augmenting the default info/warn/error)

const DEBUG = Ref{Bool}()
function debug(io::IO, msg...; prefix="DEBUG: ")
    if DEBUG[]
        Base.println_with_color(:green, io, prefix, chomp(string(msg...)))
    end
end
debug(msg...; prefix="DEBUG: ") = debug(STDERR, msg..., prefix=prefix)

const TRACE = Ref{Bool}()
function trace(io::IO, msg...; prefix="TRACE: ", line=true)
    if TRACE[]
        Base.print_with_color(:cyan, io, prefix, chomp(string(msg...)))
        if line
            println(io)
        end
    end
end
trace(msg...; prefix="TRACE: ", line=true) = trace(STDERR, msg..., prefix=prefix, line=line)


# Generate a temporary file with specific suffix
function mkstemps(suffix::AbstractString)
    b = joinpath(tempdir(), "tmpXXXXXX$suffix")
    # NOTE: mkstemps modifies b, which should be a NULL-terminated string
    p = ccall(:mkstemps, Int32, (Cstring, Cint), b, length(suffix))
    systemerror(:mktemp, p == -1)
    return (b, fdio(p, true))
end


function __init_util__()
    # IDEA: make these decisions at compile-time, avoiding runtime overhead
    TRACE[] = haskey(ENV, "TRACE")
    DEBUG[] = TRACE[] || haskey(ENV, "DEBUG")

    if TRACE[]
        trace("CUDA.jl is running in trace mode, this will generate a lot of additional output")
    elseif DEBUG[]
        debug("CUDA.jl is running in debug mode, this will generate additional output")
        debug("Run with TRACE=1 to enable even more output")
    end

    if TRACE[]
        # When in trace mode, we'll be dumping certain build artifacts to disk
        dumpdir[] = begin
            root = tempdir()

            # Find a unique directory name
            dir = ""
            i = 0
            while true
                dir = joinpath(root, "JuliaCUDA_$i")
                isdir(dir) || break
                i += 1
            end

            mkdir(dir)
            dir
        end
    end
end
