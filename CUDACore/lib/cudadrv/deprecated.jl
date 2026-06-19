# Deprecated high-level functionality
#
# Compatibility shims for high-level driver functionality that has been removed. This is
# distinct from `libcuda_deprecated.jl`, which wraps deprecated low-level CUDA C APIs.

export device_reset!

"""
    device_reset!(dev::CuDevice=device())

Reset the CUDA state associated with a device.

!!! warning

    This function is deprecated and is now a no-op. CUDA.jl no longer supports resetting a
    device: a device's primary context is kept alive for the lifetime of the process, so
    there is no longer any state to reset.
"""
function device_reset!(dev::Union{CuDevice,Nothing}=nothing)
    Base.depwarn("`device_reset!` is deprecated and is now a no-op; CUDA.jl no longer supports resetting a device.", :device_reset!)
    return
end
