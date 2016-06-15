# Utilities

const dumpdir = Ref{String}()

# Generate a temporary file with specific suffix
function mkstemps(suffix::AbstractString)
    b = joinpath(tempdir(), "tmpXXXXXX$suffix")
    # NOTE: mkstemps modifies b, which should be a NULL-terminated string
    p = ccall(:mkstemps, Int32, (Cstring, Cint), b, length(suffix))
    systemerror(:mktemp, p == -1)
    return (b, fdio(p, true))
end

function __init_util__()
    if TRACE
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
