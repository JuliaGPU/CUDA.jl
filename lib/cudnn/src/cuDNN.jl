"""
    cuDNN

High level interface to cuDNN functions. See
[README.md](https://github.com/JuliaGPU/CUDA.jl/blob/master/lib/cudnn/README.md) for a
design overview.
"""
module cuDNN

using CUDACore
using CUDACore: CUstream, CUgraph, libraryPropertyType
using CUDACore: retry_reclaim, isdebug, initialize_context, @gcsafe_ccall, @checked

using CEnum: @cenum

if CUDACore.local_toolkit
    using CUDA_Runtime_Discovery
else
    import CUDNN_jll
end


@public functional

const _initialized = Ref{Bool}(false)
functional() = _initialized[]

# core library
include("libcudnn.jl")

# low-level wrappers
include("error.jl")
include("util.jl")
include("base.jl")
include("descriptors.jl")
include("tensor.jl")
include("inplace.jl")
include("optensor.jl")
include("reduce.jl")
include("convolution.jl")
include("pooling.jl")
include("activation.jl")
include("softmax.jl")
include("dropout.jl")
include("rnn.jl")
include("multiheadattn.jl")
include("normalization.jl")


function math_mode(mode=CUDACore.math_mode())
    if mode == CUDACore.PEDANTIC_MATH
        # don't use tensor cores.
        # on A100, only use them for TF32
        CUDNN_DEFAULT_MATH
    elseif mode == CUDACore.DEFAULT_MATH
        # allow tensor core usage
        CUDNN_TENSOR_OP_MATH
    elseif mode == CUDACore.FAST_MATH
        # also downcast inputs
        CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION
    end
end


## handles

function handle_ctor(ctx)
    context!(ctx) do
        cudnnCreate()
    end
end
function handle_dtor(ctx, handle)
    context!(ctx; skip_destroyed=true) do
        cudnnDestroy(handle)
    end
end
const idle_handles = HandleCache{CuContext,cudnnHandle_t}(handle_ctor, handle_dtor)

# mutable wrapper so the raw handle is released via an object-bound
# finalizer: when TLS state is cleared on reclaim (or the owning task is
# collected) and GC runs, the wrapper is collected and its finalizer
# returns the handle to the idle cache.
mutable struct cudnnHandle
    const handle::cudnnHandle_t
    const ctx::CuContext
end
Base.unsafe_convert(::Type{cudnnHandle_t}, h::cudnnHandle) = h.handle

function handle_finalizer(h::cudnnHandle)
    push!(idle_handles, h.ctx, h.handle)
end

const LibraryState = @NamedTuple{handle::cudnnHandle, stream::CuStream}
const state_cache = CUDACore.TaskLocalCache{CuContext, LibraryState}(:cuDNN)

function handle()
    cuda = CUDACore.active_state()

    states = CUDACore.task_dict(state_cache)

    # get library state
    @noinline function new_state(cuda)
        new_handle = pop!(idle_handles, cuda.context)
        wrapped = cudnnHandle(new_handle, cuda.context)
        finalizer(handle_finalizer, wrapped)

        cudnnSetStream(new_handle, cuda.stream)

        (; handle=wrapped, cuda.stream)
    end
    state = get!(states, cuda.context) do
        new_state(cuda)
    end

    # update stream
    @noinline function update_stream(cuda, state)
        cudnnSetStream(state.handle, cuda.stream)
        (; state.handle, cuda.stream)
    end
    if state.stream != cuda.stream
        states[cuda.context] = state = update_stream(cuda, state)
    end

    return state.handle
end


## logging

function log_message(sev, udata, dbg_ptr, ptr)
    dbg = unsafe_load(dbg_ptr)

    # find the length of the message, as denoted by two null terminators
    len = 0
    while true && len < 10000
        if unsafe_load(ptr, len+1) == 0 && unsafe_load(ptr, len+2) == 0
            break
        end
        len += 1
    end
    str = unsafe_string(ptr, len)

    # split into lines and report
    lines = split(str, '\0')
    msg = join(strip.(lines), '\n')
    if sev == CUDNN_SEV_INFO
        @debug msg
    elseif sev == CUDNN_SEV_WARNING
        @warn msg
    elseif sev == CUDNN_SEV_ERROR
        @error msg
    elseif sev == CUDNN_SEV_FATAL
        error(msg)
    end

    return
end

function __init__()
    precompiling = ccall(:jl_generating_output, Cint, ()) != 0

    CUDACore.functional() || return

    # find the library
    global libcudnn
    if CUDACore.local_toolkit
        dirs = CUDA_Runtime_Discovery.find_toolkit()
        path = CUDA_Runtime_Discovery.get_library(dirs, "cudnn"; optional=true)
        if path === nothing
            precompiling || @error "cuDNN is not available on your system (looked in $(join(dirs, ", ")))"
            return
        end
        libcudnn = path
    else
        if !CUDNN_jll.is_available()
            precompiling || @error "cuDNN is not available for your platform ($(Base.BinaryPlatforms.triplet(CUDNN_jll.host_platform)))"
            return
        end
        libcudnn = CUDNN_jll.libcudnn
    end

    # register a log callback
    if !precompiling && (isdebug(:init, cuDNN) || Base.JLOptions().debug_level >= 2)
        callback = @cfunction(log_message, Nothing,
                              (cudnnSeverity_t, Ptr{Cvoid}, Ptr{cudnnDebug_t}, Ptr{UInt8}))
        cudnnSetCallback(typemax(UInt32), C_NULL, callback)
    end

    CUDACore.register_reclaimable!(idle_handles)
    CUDACore.register_reclaimable!(state_cache)

    _initialized[] = true
end

include("precompile.jl")

end
