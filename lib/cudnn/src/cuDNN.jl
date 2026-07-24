"""
    cuDNN

High level interface to cuDNN functions. See
[README.md](https://github.com/JuliaGPU/CUDA.jl/blob/main/lib/cudnn/README.md) for a
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
include("backend.jl")
include("graph/graph.jl")
include("graph/lower.jl")
include("graph/ops.jl")
include("graph/engines.jl")
include("graph/execute.jl")
include("graph/cache.jl")
include("ops/attention.jl")
include("ops/convolution.jl")
include("ops/pooling.jl")
include("ops/normalization.jl")

# fixed-function compatibility wrappers
include("legacy/descriptors.jl")
include("legacy/inplace.jl")
include("legacy/optensor.jl")
include("legacy/reduce.jl")
include("legacy/convolution.jl")
include("legacy/pooling.jl")
include("legacy/activation.jl")
include("legacy/multiheadattn.jl")
include("legacy/normalization.jl")

include("dropout.jl")
include("rnn.jl")
include("softmax.jl")


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

# Execution plans are bound to their cuDNN handle.
struct PooledHandle
    handle::cudnnHandle_t
    plans::Dict{Any,Any}
end

function handle_ctor(ctx)
    context!(ctx) do
        PooledHandle(cudnnCreate(), Dict{Any,Any}())
    end
end
function handle_dtor(ctx, pooled::PooledHandle)
    context!(ctx) do
        for plan in values(pooled.plans)
            plan isa Graph && unsafe_destroy!(plan)
        end
        empty!(pooled.plans)
        cudnnDestroy(pooled.handle)
    end
end
const idle_handles = HandleCache{CuContext,PooledHandle}(handle_ctor, handle_dtor)

# mutable wrapper so the raw handle is released via an object-bound
# finalizer: when TLS state is cleared on reclaim (or the owning task is
# collected) and GC runs, the wrapper is collected and its finalizer
# returns the handle to the idle cache.
mutable struct Handle
    const handle::cudnnHandle_t
    const ctx::CuContext
    const plans::Dict{Any,Any}
end
Base.unsafe_convert(::Type{cudnnHandle_t}, h::Handle) = h.handle

function handle_finalizer(h::Handle)
    push!(idle_handles, h.ctx, PooledHandle(h.handle, h.plans))
end

const LibraryState = @NamedTuple{handle::Handle, stream::CuStream}
const state_cache = CUDACore.TaskLocalCache{CuContext, LibraryState}(:cuDNN)

function handle()
    cuda = CUDACore.active_state()

    states = CUDACore.task_dict(state_cache)

    # get library state
    @noinline function new_state(cuda)
        pooled = pop!(idle_handles, cuda.context)
        wrapped = Handle(pooled.handle, cuda.context, pooled.plans)
        finalizer(handle_finalizer, wrapped)

        cudnnSetStream(pooled.handle, cuda.stream)

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
