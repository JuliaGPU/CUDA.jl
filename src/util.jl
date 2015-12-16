export
    mkstemps

const logger = Ref{Logger}()
const dumpdir = Ref{ASCIIString}()

# Generate a temporary file with specific suffix
function mkstemps(suffix::AbstractString)
    b = joinpath(tempdir(), "tmpXXXXXX$suffix")
    # NOTE: mkstemps modifies b, which should be a NULL-terminated string
    p = ccall(:mkstemps, Int32, (Cstring, Cint), b, length(suffix))
    systemerror(:mktemp, p == -1)
    return (b, fdio(p, true))
end

function __init_util__()
    logger[] = Logger("CUDA.jl", haskey(ENV, "DEBUG") ? DEBUG : WARNING, STDERR)

    # Logging shadows info(), so let it work as intended again
    Logging.configure(output=STDOUT, level=Logging.INFO)

    # When in debug mode, we dump certain artifacts for inspection purposes.
    if logger[].level <= DEBUG
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
