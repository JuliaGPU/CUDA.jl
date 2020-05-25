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
    else
        @warn("""Calling CUDA.@profile only informs an external profiler to start.
                 The user is responsible for launching Julia under a CUDA profiler like `nvprof`.

                 For improved usability, launch Julia under the Nsight Systems profiler:
                 \$ nsys launch -t cuda,cublas,cudnn,nvtx julia""",
              maxlog=1)
    end
    CUDA.cuProfilerStart()
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
    @warn "Precompiling while running under Nsight Systems; make sure you disable OSRT tracing to prevent segmentation faults (exit code 11)"
end

function __init__()
    # find the active Nsight Systems profiler
    if haskey(ENV, "NSYS_PROFILING_SESSION_ID") && ccall(:jl_generating_output, Cint, ()) == 0
        @info "Running under Nsight Systems, CUDA.@profile will automatically start the profiler"

        nsight[] = ENV["_"]
        @assert isfile(nsight[])
    else
        nsight[] = nothing
    end
end

end
