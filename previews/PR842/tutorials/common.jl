# function to run a Julia script outside of the current environment
function script(code; wrapper=``, args=``)
    if Base.JLOptions().project != C_NULL
        args = `$args --project=$(unsafe_string(Base.JLOptions().project))`
    end
    mktemp() do path, io
        write(io, code)
        flush(io)
        cmd = `$wrapper $(Base.julia_cmd()) $args $path`
        # redirect stderr to stdout to have it picked up by Weave.jl
        run(pipeline(ignorestatus(cmd), stderr=stdout))
    end
    nothing
end
