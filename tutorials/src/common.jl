# function to run a Julia script outside of the current environment
function script(code; wrapper=``, args=``)
    mktemp() do path, io
        write(io, code)
        flush(io)
        withenv("JULIA_LOAD_PATH" => join(LOAD_PATH, ':')) do
            cmd = `$wrapper $(Base.julia_cmd()) $path $args`
            # redirect stderr to stdout to have it picked up by Weave.jl
            run(pipeline(ignorestatus(cmd), stderr=stdout))
        end
    end
    nothing
end
