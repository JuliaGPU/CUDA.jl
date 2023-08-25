module cuTENSOR

using CUDA
using CUDA.APIUtils
using CUDA: CUstream, cudaDataType
using CUDA: retry_reclaim, initialize_context, isdebug

using CEnum: @cenum

if CUDA.local_toolkit
    using CUDA_Runtime_Discovery
else
    import CUTENSOR_jll
end


export has_cutensor

const _initialized = Ref{Bool}(false)
has_cutensor() = _initialized[]


const cudaDataType_t = cudaDataType

# core library
include("libcutensor.jl")

# low-level wrappers
include("error.jl")
include("tensor.jl")
include("wrappers.jl")

# high-level integrations
include("interfaces.jl")

# cache for created, but unused handles
const idle_handles = HandleCache{CuContext,Ptr{cutensorHandle_t}}()

function handle()
    cuda = CUDA.active_state()

    # every task maintains library state per device
    LibraryState = @NamedTuple{handle::Ptr{cutensorHandle_t}}
    states = get!(task_local_storage(), :cuTENSOR) do
        Dict{CuContext,LibraryState}()
    end::Dict{CuContext,LibraryState}

    # get library state
    @noinline function new_state(cuda)
        new_handle = pop!(idle_handles, cuda.context) do
            cutensorCreate()
        end

        finalizer(current_task()) do task
            push!(idle_handles, cuda.context, new_handle) do
                cutensorDestroy(new_handle)
            end
        end

        (; handle=new_handle)
    end
    state = get!(states, cuda.context) do
        new_state(cuda)
    end

    return state.handle
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
    precompiling = ccall(:jl_generating_output, Cint, ()) != 0

    CUDA.functional() || return

    # find the library
    global libcutensor
    if CUDA.local_toolkit
        dirs = CUDA_Runtime.find_toolkit()
        path = CUDA_Runtime.get_library(dirs, "cutensor"; optional=true)
        if path === nothing
            precompiling || @error "cuTENSOR is not available on your system (looked in $(join(dirs, ", ")))"
            return
        end
        libcutensor = path
    else
        if !CUTENSOR_jll.is_available()
            precompiling || @error "cuTENSOR is not available for your platform ($(Base.BinaryPlatforms.triplet(CUTENSOR_jll.host_platform)))"
            return
        end
        libcutensor = CUTENSOR_jll.libcutensor
    end

    # register a log callback
    if isdebug(:init, cuTENSOR) || Base.JLOptions().debug_level >= 2
        callback = @cfunction(log_message, Nothing, (Int32, Cstring, Cstring))
        cutensorLoggerSetCallback(callback)
        cutensorLoggerOpenFile(Sys.iswindows() ? "NUL" : "/dev/null")
        cutensorLoggerSetLevel(5)
    end

    _initialized[] = true
end

end
