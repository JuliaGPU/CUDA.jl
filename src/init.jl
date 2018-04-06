# Initialization

export device!


const initialized = Ref{Bool}(false)
const device_contexts = Dict{CuDevice,CuContext}()

# FIXME: support for flags (see `cudaSetDeviceFlags`)

const device_apicalls = Set{Symbol}([
    # these API calls should not initialize the API,
    # as they are commonly used to determine the most appropriate device
    :cuDeviceGet,
    :cuDeviceGetAttribute,
    :cuDeviceGetCount,
    :cuDeviceGetName,
    :cuDeviceTotalMem
])

function init_device(apicall)
    apicall in device_apicalls && return

    # NOTE: we could do something smarter here,
    #       eg. select the most powerful device,
    #       or skip devices without free memory
    dev = CuDevice(0)
    @debug "Initializing CUDA for device 0 after API call" name=CUDAdrv.name(dev) apicall
    device!(dev)
end

"""
    device!(dev)

Sets `dev` as the current active device for the calling host thread. Devices can be
specified by integer id, or as a `CuDevice`. This is intended to be a low-cost operation,
only performing significant work when calling it for the first time for each device.
"""
function device!(dev::CuDevice)
    if !initialized[]
        initialized[] = true
        CUDAdrv.apicall_hook[] = nothing
    end

    # NOTE: although these conceptually match what the primary context is for,
    #       we don't use that because it is refcounted separately
    #       and might confuse / be confused by user operations
    #       (eg. calling `unsafe_reset!` on a primary context)
    if haskey(device_contexts, dev)
        ctx = device_contexts[dev]
        activate(ctx)
    else
        device_contexts[dev] = CuContext(dev)
    end
end
device!(dev::Integer) = device!(CuDevice(dev))

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
        isnull(old_ctx) || activate(old_ctx)
    end
end
device!(f::Function, dev::Integer) = device!(f, CuDevice(dev))

const jlctx = Ref{LLVM.Context}()
function __init__()
    jlctx[] = LLVM.Context(convert(LLVM.API.LLVMContextRef,
                                   cglobal(:jl_LLVMContext, Cvoid)))

    if !configured
        @warn """CUDAnative.jl has not been successfully built, and will not work properly.
                 Please run Pkg.build(\"CUDAnative\") and restart Julia."""
        return
    end

    if CUDAdrv.version() != cuda_driver_version
        error("Your set-up has changed. Please run Pkg.build(\"CUDAnative\") and restart Julia.")
    end

    CUDAdrv.apicall_hook[] = init_device
    init_jit()
end
