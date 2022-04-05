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
        try
            $(esc(ex))
        finally
            Profile.stop()
        end
    end
end


module Profile

using ..CUDA

function find_nsys()
    if haskey(ENV, "JULIA_CUDA_NSYS")
        return ENV["JULIA_CUDA_NSYS"]
    elseif haskey(ENV, "_") && contains(ENV["_"], r"nsys"i)
        # NOTE: if running as e.g. Jupyter -> nsys -> Julia, _ is `jupyter`
        return ENV["_"]
    else
        # look at a couple of environment variables that may point to NSight
        nsight = nothing
        for var in ("LD_PRELOAD", "CUDA_INJECTION64_PATH", "NVTX_INJECTION64_PATH")
            haskey(ENV, var) || continue
            for val in split(ENV[var], Sys.iswindows() ? ';' : ':')
                isfile(val) || continue
                candidate = if Sys.iswindows()
                    joinpath(dirname(val), "nsys.exe")
                else
                    joinpath(dirname(val), "nsys")
                end
                isfile(candidate) && return candidate
            end
        end
    end
    error("Running under Nsight Systems, but could not find the `nsys` binary to start the profiler. Please specify using JULIA_CUDA_NSYS=path/to/nsys, and file an issue with the contents of ENV.")
end

const __nsight = Ref{Union{Nothing,String}}()
function nsight()
    if !isassigned(__nsight)
        # find the active Nsight Systems profiler
        if haskey(ENV, "NSYS_PROFILING_SESSION_ID") && ccall(:jl_generating_output, Cint, ()) == 0
            __nsight[] = find_nsys()
            @assert isfile(__nsight[])
            @info "Running under Nsight Systems, CUDA.@profile will automatically start the profiler"
        else
            __nsight[] = nothing
        end
    end

    __nsight[]
end


"""
    start()

Enables profile collection by the active profiling tool for the current context. If
profiling is already enabled, then this call has no effect.
"""
function start()
    if nsight() !== nothing
        run(`$(nsight()) start --capture-range=cudaProfilerApi`)
        # it takes a while for the profiler to actually start tracing our process
        sleep(0.01)
    else
        @warn("""Calling CUDA.@profile only informs an external profiler to start.
                 The user is responsible for launching Julia under a CUDA profiler.

                 It is recommended to use Nsight Systems, which supports interactive profiling:
                 \$ nsys launch julia""",
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
    CUDA.cuProfilerStop()
    if nsight() !== nothing
        @info """Profiling has finished, open the report listed above with `nsys-ui`
                 If no report was generated, try launching `nsys` with `--trace=cuda`"""
    end
end

end
