# Profiler control

export
    @profile, @cuprofile

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

using ..CUDAdrv
using Distributed, Libdl

const nsight = Ref{Union{Nothing,String}}(nothing)

# use a separate process to communicate with nsight to work around
# https://devtalk.nvidia.com/default/topic/1065305/
#
# note that other process spawning is still broken, and e.g. precompilation does not work
const child = Ref{Int}()


"""
    start()

Enables profile collection by the active profiling tool for the current context. If
profiling is already enabled, then this call has no effect.
"""
function start()
    # make sure the stop call is precompiled
    stop()

    if nsight[] !== nothing
        remotecall_wait(child[], nsight[]) do nsight
            run(`$nsight start -c cudaProfilerApi`)
        end
    else
        @warn("""Calling CUDAdrv.@profile only informs an external profiler to start.
                 The user is responsible for launching Julia under a CUDA profiler like `nvprof`.

                 For improved usability, launch Julia under the Nsight Systems profiler.""",
              maxlog=1)
    end
    CUDAdrv.cuProfilerStart()
end

"""
    stop()

Disables profile collection by the active profiling tool for the current context. If
profiling is already disabled, then this call has no effect.
"""
stop() = CUDAdrv.cuProfilerStop()

function __init__()
    # find the active Nsight Systems profiler
    libs = Libdl.dllist()
    filter!(path->occursin("ToolsInjection", basename(path)), libs)
    if isempty(libs)
        nsight[] = nothing
    else
        dirs = unique(map(dirname, libs))
        @assert length(dirs) == 1
        dir = dirs[1]

        nsight[] = joinpath(dir, "nsys")
        @assert isfile(nsight[])

        child[] = withenv("LD_PRELOAD" => nothing) do
            addprocs(1)[1]
        end

        @info "Running under Nsight Systems, CUDAdrv.@profile will automatically start the profiler"
    end
end

end
