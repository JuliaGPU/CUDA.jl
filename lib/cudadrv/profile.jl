# Profiler control

"""
    @profile ex

Run expressions while activating the CUDA profiler.

Note that this API is used to programmatically control the profiling granularity by allowing
profiling to be done only on selective pieces of code. It does not perform any profiling on
itself, you need external tools for that.
"""
macro profile(ex)
    quote
        Profile.start()
        local ret = $(esc(ex))
        Profile.stop()
        ret
    end
end


module Profile

using ..CUDA

const nsight = Ref{Union{Nothing,String}}(nothing)


"""
    start()

Enables profile collection by the active profiling tool for the current context. If
profiling is already enabled, then this call has no effect.
"""
function start()
    if nsight[] !== nothing
        run(`$(nsight[]) start`)
        # it takes a while for the profiler to actually start tracing our process
        sleep(0.01)
    else
        @warn("""Calling CUDA.@profile only informs an external profiler to start.
                 The user is responsible for launching Julia under a CUDA profiler.

                 It is recommended to use Nsight Systems, which supports interactive profiling:
                 \$ nsys launch julia""",
              maxlog=1)
        CUDA.cuProfilerStart()
    end
end

"""
    stop()

Disables profile collection by the active profiling tool for the current context. If
profiling is already disabled, then this call has no effect.
"""
function stop()
    if nsight[] !== nothing
        run(`$(nsight[]) stop`)
        @info "Profiling has finished, open the report listed above with `nsight-sys`"
    else
        CUDA.cuProfilerStop()
    end
end

if haskey(ENV, "NSYS_PROFILING_SESSION_ID") && ccall(:jl_generating_output, Cint, ()) == 1
    @warn "Precompiling while running under Nsight Systems; if you encounter segmentation faults (exit code 11), try disabling OSRT tracing."
end

function __init__()
    # find the active Nsight Systems profiler
    if haskey(ENV, "NSYS_PROFILING_SESSION_ID") && ccall(:jl_generating_output, Cint, ()) == 0
        nsight[] = if haskey(ENV, "JULIA_CUDA_NSYS")
            ENV["JULIA_CUDA_NSYS"]
        elseif haskey(ENV, "_")
            ENV["_"]
        elseif haskey(ENV, "LD_PRELOAD")
            libraries = split(ENV["LD_PRELOAD"], ':')
            filter!(isfile, libraries)
            directories = map(dirname, libraries)
            candidates = map(dir->joinpath(dir, "nsys"), directories)
            filter!(isfile, candidates)
            isempty(candidates) && error("Could not find nsys relative to LD_PRELOAD=$(ENV["LD_PRELOAD"])")
            first(candidates)
        else
            error("Running under Nsight Systems, but could not find the `nsys` binary to start the profiler. Please specify using JULIA_CUDA_NSYS=path/to/nsys, and file an issue with the contents of ENV.")
        end
        @assert isfile(nsight[])
        @info "Running under Nsight Systems, CUDA.@profile will automatically start the profiler"
    else
        nsight[] = nothing
    end
end

end
