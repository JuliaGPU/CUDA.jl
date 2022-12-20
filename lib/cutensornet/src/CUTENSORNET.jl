module CUTENSORNET

using LinearAlgebra
using CUDA
using CUDA: CUstream, cudaDataType, @checked, HandleCache, with_workspace
using CUDA: @retry_reclaim, initialize_context, isdebug

using CUTENSOR
using CUTENSOR: CuTensor

using CEnum: @cenum

using cuQuantum_jll


export has_cutensornet

function has_cutensornet(show_reason::Bool=false)
    if !isdefined(cuQuantum_jll, :libcutensornet)
        show_reason && error("cuTensorNet library not found")
        return false
    end
    return true
end


const cudaDataType_t = cudaDataType

# core library
include("libcutensornet.jl")

include("error.jl")
include("types.jl")
include("tensornet.jl")

# cache for created, but unused handles
const idle_handles = HandleCache{CuContext,cutensornetHandle_t}()

function handle()
    cuda = CUDA.active_state()

    # every task maintains library state per device
    LibraryState = @NamedTuple{handle::cutensornetHandle_t}
    states = get!(task_local_storage(), :CUTENSORNET) do
        Dict{CuContext,LibraryState}()
    end::Dict{CuContext,LibraryState}

    # get library state
    @noinline function new_state(cuda)
        new_handle = pop!(idle_handles, cuda.context) do
            handle = Ref{cutensornetHandle_t}()
            cutensornetCreate(handle)
            handle[]
        end

        finalizer(current_task()) do task
            push!(idle_handles, cuda.context, new_handle) do
                context!(cuda.context; skip_destroyed=true) do
                    cutensornetDestroy(new_handle)
                end
            end
        end

        (; handle=new_handle)
    end
    state = get!(states, cuda.context) do
        new_state(cuda)
    end

    return state.handle
end

function version()
  ver = cutensornetGetVersion()
  major, ver = divrem(ver, 10000)
  minor, patch = divrem(ver, 100)

  VersionNumber(major, minor, patch)
end

function cuda_version()
  ver = cutensornetGetCudartVersion()
  major, ver = divrem(ver, 1000)
  minor, patch = divrem(ver, 10)

  VersionNumber(major, minor, patch)
end


## logging

function log_message(log_level, function_name, message)
    function_name = unsafe_string(function_name)
    message = unsafe_string(message)
    output = if isempty(message)
        "$function_name(...)"
    else
        "$function_name: $message"
    end
    if log_level <= 1
        @error output
    else
        # the other log levels are different levels of tracing and hints
        @debug output
    end
    return
end

function __init__()
    # register a log callback
    if isdebug(:init, CUTENSORNET) || Base.JLOptions().debug_level >= 2
        callback = @cfunction(log_message, Nothing, (Int32, Cstring, Cstring))
        cutensornetLoggerSetCallback(callback)
        cutensornetLoggerOpenFile(Sys.iswindows() ? "NUL" : "/dev/null")
        cutensornetLoggerSetLevel(5)
    end
end

end
