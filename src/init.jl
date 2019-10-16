# Initialization

export device!, device_reset!


const initialized = Ref{Bool}(false)
const device_contexts = Dict{CuDevice,CuContext}()

# FIXME: support for flags (see `cudaSetDeviceFlags`)

# API calls that are allowed without lazily initializing the CUDA library
#
# this list isn't meant to be complete (ie. many other API calls are actually allowed
# without setting-up a context), and only serves to make multi-device applications possible.
#
# feel free to open a PR adding additional API calls, if you have a specific use for them.
const preinit_apicalls = Set{Symbol}([
    :cuDriverGetVersion,
    # device calls, commonly used to determine the most appropriate device
    :cuDeviceGet,
    :cuDeviceGetAttribute,
    :cuDeviceGetCount,
    :cuDeviceGetName,
    :cuDeviceTotalMem,
    # context calls, for testing
    :cuCtxGetCurrent
])

function maybe_initialize(apicall)
    initialized[] && return
    apicall in preinit_apicalls && return
    @debug "Initializing CUDA after call to $apicall"
    initialize()
end

function initialize(dev = CuDevice(0))
    # NOTE: we could do something smarter here,
    #       eg. select the most powerful device,
    #       or skip devices without free memory
    device!(dev)
end

const device!_listeners = Set{Function}()

"""
    device!(dev)

Sets `dev` as the current active device for the calling host thread. Devices can be
specified by integer id, or as a `CuDevice`. This is intended to be a low-cost operation,
only performing significant work when calling it for the first time for each device.

If your library or code needs to perform an action when the active device changes, add a
callback of the signature `(::CuDevice, ::CuContext)` to the `device!_listeners` set.
"""
function device!(dev::CuDevice)
    if !initialized[]
        initialized[] = true
        CUDAdrv.apicall_hook[] = nothing
    end

    ctx = get!(device_contexts, dev) do
        pctx = CuPrimaryContext(dev)
        CuContext(pctx)
        pop!(CuContext) # we don't maintain a stack...
    end
    activate(ctx)       # replace the top of the stack

    for listener in device!_listeners
        listener(dev, device_contexts[dev])
    end
end
device!(dev::Integer) = device!(CuDevice(dev))

const device_reset!_listeners = Set{Function}()

"""
    device_reset!(dev::CuDevice=device())

Reset the CUDA state associated with a device. This call with release the underlying
context, at which point any objects allocated in that context will be invalidated.

If your library or code needs to perform an action when a device is resetted, add a
callback of the signature `(::CuDevice, ::CuContext)` to the `device_reset!_listeners` set.
"""
function device_reset!(dev::CuDevice=device())
    active_ctx = CuCurrentContext()
    if active_ctx == nothing
        active_dev = nothing
    else
        active_dev = device(active_ctx)
    end

    if haskey(device_contexts, dev)
        for listener in device_reset!_listeners
            listener(dev, device_contexts[dev])
        end
        delete!(device_contexts, dev)
    end

    # unconditionally reset the primary context,
    # as there might be users outside of CUDAnative.jl
    pctx = CuPrimaryContext(dev)
    unsafe_reset!(pctx)

    if dev == active_dev
        # unless the user switches devices, new API calls should trigger initialization
        CUDAdrv.apicall_hook[] = maybe_initialize
        initialized[] = false
    end

    return
end

"""
    device!(f, dev)

Sets the active device for the duration of `f`.
"""
function device!(f::Function, dev::CuDevice)
    # FIXME: should use Push/Pop
    old_ctx = CuCurrentContext()
    try
        device!(dev)
        f()
    finally
        if old_ctx != nothing
            activate(old_ctx)
        end
    end
end
device!(f::Function, dev::Integer) = device!(f, CuDevice(dev))
