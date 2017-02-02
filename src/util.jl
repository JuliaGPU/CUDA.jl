# Utilities

const dumpdir = Ref{String}()

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
