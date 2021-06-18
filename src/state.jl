# global state management
#
# The functionality in this file serves to create a coherent environment to perform
# computations in, with support for Julia constructs like tasks (and executing those on
# multiple threads), using a GPU of your choice (with the ability to reset that device, or
# use different devices on different threads).
#
# this is complicated by the fact that CUDA uses thread-bound contexts, while we want to be
# able to switch between tasks executing on the same thread. the current approach involves
# prefixing every CUDA API call with code that ensures the CUDA thread-bound state is
# identical to Julia's task-local one.

export context, context!, device, device!, device_reset!, deviceid, stream, stream!


## task-local state

@enum MathMode begin
    # use prescribed precision and standardized arithmetic for all calculations.
    # this may serialize operations, and reduce performance.
    PEDANTIC_MATH

    # use at least the required precision, and allow reordering operations for performance.
    DEFAULT_MATH

    # additionally allow downcasting operations for better use of hardware resources.
    # whenever possible the `precision` flag passed to `math_mode!` will be used
    # to constrain those downcasts.
    FAST_MATH
end

# math mode and precision are sticky (once set on a task, inherit to newly created tasks)
const default_math_mode = Ref{Union{Nothing,MathMode}}(nothing)
const default_math_precision = Ref{Union{Nothing,Symbol}}(nothing)

# the default device unitialized tasks will use, set when switching devices.
# this behavior differs from the CUDA Runtime, where device 0 is always used.
# this setting won't be used when switching tasks on a pre-initialized thread.
const default_device = Ref{Union{Nothing,CuDevice}}(nothing)

mutable struct TaskLocalState
    device::CuDevice
    context::CuContext
    streams::Vector{Union{Nothing,CuStream}}
    math_mode::MathMode
    math_precision::Symbol

    function TaskLocalState(dev::CuDevice=something(default_device[], CuDevice(0)),
                            ctx::CuContext = context(dev))
        math_mode = something(default_math_mode[],
                              Base.JLOptions().fast_math==1 ? FAST_MATH : DEFAULT_MATH)
        math_precision = something(default_math_precision[], :TensorFloat32)
        new(dev, ctx, Base.fill(nothing, ndevices()), math_mode, math_precision)
    end
end

function validate_task_local_state(state::TaskLocalState)
    # NOTE: the context may be invalid if another task reset it (which we detect here
    #       since we can't touch other tasks' local state from `device_reset!`)
    if !isvalid(state.context)
        device!(state.device)
        state.streams[deviceid(state.device)+1] = nothing
    end
    return state
end

# get or create the task local state, and make sure it's valid
function task_local_state!(args...)
    tls = task_local_storage()
    if haskey(tls, :CUDA)
        validate_task_local_state(@inbounds(tls[:CUDA]))
    else
        tls[:CUDA] = TaskLocalState(args...)
    end::TaskLocalState
end

# only get the task local state (it may be invalid!), or return nothing if unitialized
function task_local_state()
    tls = task_local_storage()
    if haskey(tls, :CUDA)
        @inbounds(tls[:CUDA])
    else
        nothing
    end::Union{TaskLocalState,Nothing}
end

@inline function prepare_cuda_state()
    state = task_local_state!()

    # NOTE: CuCurrentContext() is too slow to use here (taking a lock, accessing a dict)
    #       so we use the raw handle. is that safe though, when we reset the device?
    #ctx = CuCurrentContext()
    ctx = Ref{CUcontext}()
    cuCtxGetCurrent(ctx)
    if ctx[] != state.context.handle
        activate(state.context)
    end

    return
end

# convenience function to get all relevant state
# without querying task local storage multiple times
@inline function active_state()
    # inline to remove unused state properties
    state = task_local_state!()
    return (device=state.device, context=state.context, stream=stream(state),
            math_mode=state.math_mode, math_precision=state.math_precision)
end


## context-based API

"""
    context()::CuContext

Get or create a CUDA context for the current thread (as opposed to
`CuCurrentContext` which may return `nothing` if there is no context bound to the
current thread).
"""
function context()
    task_local_state!().context
end

"""
    context!(ctx::CuContext)

Bind the current host thread to the context `ctx`. Returns the previously-bound context.

Note that the contexts used with this call should be previously acquired by calling
[`context`](@ref), and not arbitrary contexts created by calling the `CuContext` constructor.
"""
function context!(ctx::CuContext)
    activate(ctx) # we generally only apply CUDA state lazily, i.e. in `prepare_cuda_state`,
                    # but we need to do so early here to be able to get the context's device.
    dev = CuCurrentDevice()::CuDevice

    # switch contexts
    state = task_local_state()
    if state === nothing
        old_ctx = nothing
        task_local_state!(dev, ctx)
    else
        old_ctx = state.context
        state.device = dev
        state.context = ctx
    end

    return old_ctx
end

macro context!(ex...)
    body = ex[end]
    ctx = ex[end-1]
    kwargs = ex[1:end-2]

    skip_destroyed = false
    for kwarg in kwargs
        Meta.isexpr(kwarg, :(=)) || throw(ArgumentError("non-keyword argument like option '$kwarg'"))
        key, val = kwarg.args
        isa(key, Symbol) || throw(ArgumentError("non-symbolic keyword '$key'"))

        if key == :skip_destroyed
            skip_destroyed = val
        else
            throw(ArgumentError("unrecognized keyword argument '$kwarg'"))
        end
    end

    quote
        ctx = $(esc(ctx))
        if isvalid(ctx)
            old_ctx = context!(ctx)
            try
                $(esc(body))
            finally
                if old_ctx !== nothing && old_ctx != ctx && isvalid(old_ctx)
                    context!(old_ctx)
                end
            end
        elseif !$(esc(skip_destroyed))
            error("Cannot switch to an invalidated context.")
        end
    end
end

"""
    context!(f, ctx; [skip_destroyed=false])

Sets the active context for the duration of `f`.
"""
@inline function context!(f::Function, ctx::CuContext; skip_destroyed::Bool=false)
    # @inline so that the kwarg method is inlined too and we can const-prop skip_destroyed
    @context! skip_destroyed=skip_destroyed ctx f()
end


## device-based API

"""
    device()::CuDevice

Get the CUDA device for the current thread, similar to how [`context()`](@ref) works
compared to [`CuCurrentContext()`](@ref).
"""
function device()
    task_local_state!().device
end

const __device_contexts = LazyInitialized{Vector{Union{Nothing,CuContext}}}() do
    [nothing for _ in 1:ndevices()]
end
function device_context(i)
    contexts = __device_contexts[]
    @inbounds contexts[i]
end
function device_context!(i, ctx)
    contexts = __device_contexts[]
    @inbounds contexts[i] = ctx
    return
end

function context(dev::CuDevice)
    devidx = deviceid(dev)+1

    # querying the primary context for a device is expensive (~100ns), so cache it
    ctx = device_context(devidx)
    if ctx !== nothing
        return ctx
    end

    if capability(dev) < v"3.5"
        @warn("""Your $(name(dev)) GPU does not meet the minimal required compute capability ($(capability(dev)) < 3.5).
                 Some functionality might be unavailable.""",
              maxlog=1, _id=devidx)
    end

    # configure the primary context
    pctx = CuPrimaryContext(dev)
    ctx = CuContext(pctx)
    device_context!(devidx, ctx)
    return ctx
end

"""
    device!(dev::Integer)
    device!(dev::CuDevice)

Sets `dev` as the current active device for the calling host thread. Devices can be
specified by integer id, or as a `CuDevice` (slightly faster).
"""
function device!(dev::CuDevice, flags=nothing)
    # configure the primary context flags
    if flags !== nothing
        devidx = deviceid(dev)+1
        if device_context(devidx) !== nothing
            error("Cannot set flags for an active device. Do so before calling any CUDA function, or reset the device first.")
        end
        cuDevicePrimaryCtxSetFlags(dev, flags)
    end

    # make this device the new default
    default_device[] = dev

    # switch contexts
    state = task_local_state()
    if state === nothing
        task_local_state!(dev)
    else
        state.device = dev
        state.context = context(dev)
    end
end

macro device!(dev, body)
    quote
        ctx = context($(esc(dev)))
        @context! ctx $(esc(body))
    end
end

"""
    device!(f, dev)

Sets the active device for the duration of `f`.

Note that this call is intended for temporarily switching devices, and does not change the
default device used to initialize new threads or tasks.
"""
function device!(f::Function, dev::CuDevice)
    @device! dev f()
end

# NVIDIA bug #3240770
can_reset_device() = !(release() == v"11.2" && any(dev->pools[dev].stream_ordered, devices()))

"""
    device_reset!(dev::CuDevice=device())

Reset the CUDA state associated with a device. This call with release the underlying
context, at which point any objects allocated in that context will be invalidated.

!!! warning

    This function is broken on CUDA 11.2 when using the CUDA memory pool (the default).
    If you need to reset the device, use another memory pool by setting the
    `JULIA_CUDA_MEMORY_POOL` environment variable to, e.g., `binned` before importing
    this package.
"""
function device_reset!(dev::CuDevice=device())
    if !can_reset_device()
        @error """Due to a bug in CUDA, resetting the device is not possible on CUDA 11.2 when using the stream-ordered memory allocator.

                  If you are calling this function to free memory, that may not be required anymore
                  as the stream-ordered memory allocator releases memory much more eagerly.
                  If you do need this functionality, switch to another memory pool by setting
                  the `JULIA_CUDA_MEMORY_POOL` environment variable to, e.g., `binned`."""
        return
    end

    # unconditionally reset the primary context (don't just release it),
    # as there might be users outside of CUDA.jl
    pctx = CuPrimaryContext(dev)
    unsafe_reset!(pctx)

    # wipe the device-specific state
    devidx = deviceid(dev)+1
    device_context!(devidx, nothing)

    return
end


## integer device-based API

device!(dev::Integer, flags=nothing) = device!(CuDevice(dev), flags)
device!(f::Function, dev::Integer) = device!(f, CuDevice(dev))

"""
    deviceid()::Int
    deviceid(dev::CuDevice)::Int

Get the ID number of the current device of execution. This is a 0-indexed number,
corresponding to the device ID as known to CUDA.
"""
deviceid(dev::CuDevice=device()) = Int(convert(CUdevice, dev))


## helpers

"""
    PerDevice{T} do dev
        # generate a value of type `T` for device `dev`
    end

A helper struct for maintaining per-device state that's lazily initialized and automatically
invalidated when the device is reset. Use `per_device[dev::CuDevice]` to fetch a value.

Mutating or deleting state is not supported. If this is required, use a `Ref` as value.

Furthermore, even though the initialization of this helper, fetching its value for a
given device, and clearing it when the device is reset are all performed in a thread-safe
manner, you should still take care about thread-safety when using the contained value.
For example, if you need to update the value, use atomics; if it's a complex structure like
an array or a dictionary, use additional locks.
"""
struct PerDevice{T,F,L,C}
    lock::ReentrantLock
    constructor::F
    values::L
    contexts::C
end

function PerDevice{T}(constructor::F) where {T,F}
    values = LazyInitialized{Vector{Union{Nothing,T}}}() do
        [nothing for _ in 1:ndevices()]
    end
    contexts = LazyInitialized{Vector{Union{Nothing,CuContext}}}() do
        [nothing for _ in 1:ndevices()]
    end
    PerDevice{T,F,typeof(values),typeof(contexts)}(ReentrantLock(), constructor, values, contexts)
end

function Base.getindex(x::PerDevice, dev::CuDevice)
    id = deviceid(dev)+1
    ctx = device_context(id)    # may be nothing
    values = x.values[]
    contexts = x.contexts[]
    @inbounds begin
        # test-lock-test
        if values[id] === nothing || contexts[id] !== ctx
            Base.@lock x.lock begin
                if values[id] === nothing || contexts[id] !== ctx
                    values[id] = x.constructor(dev)
                    contexts[id] = ctx
                end
            end
        end
        values[id]
    end
end

Base.length(x::PerDevice) = length(x.values[])
Base.size(x::PerDevice) = size(x.values[])
Base.keys(x::PerDevice) = keys(x.values[])

function Base.show(io::IO, mime::MIME"text/plain", x::PerDevice{T}) where {T}
    print(io, "PerDevice{$T} with $(length(x)) entries")
end


## math mode

function math_mode!(mode::MathMode; precision=nothing)
    state = task_local_state!()

    state.math_mode = mode
    default_math_mode[] = mode

    if precision !== nothing
        state.math_precision = math_precision
        default_math_precision[] = precision
    end

    return
end

math_mode() = task_local_state!().math_mode
math_precision() = task_local_state!().math_precision


## streams

"""
    stream()

Get the CUDA stream that should be used as the default one for the currently executing task.
"""
@inline function stream(state=task_local_state!())
    # @inline so that it can be DCE'd when unused from active_state
    devidx = deviceid(state.device)+1
    if state.streams[devidx] === nothing
        stream = CuStream()

        # register the name of this task
        t = current_task()
        tptr = pointer_from_objref(current_task())
        tptrstr = string(convert(UInt, tptr), base=16, pad=Sys.WORD_SIZE>>2)
        NVTX.nvtxNameCuStreamA(stream, "Task(0x$tptrstr)")

        state.streams[devidx] = stream
    else
        state.streams[devidx]::CuStream
    end
end

function stream!(stream::CuStream)
    state = task_local_state!()
    devidx = deviceid(state.device)+1
    state.streams[devidx] = stream
    return
end

function stream!(f::Function, stream::CuStream)
    state = task_local_state!()
    devidx = deviceid(state.device)+1
    old_stream = state.streams[devidx]
    state.streams[devidx] = stream
    try
        f()
    finally
        state.streams[devidx] = old_stream
    end
end
