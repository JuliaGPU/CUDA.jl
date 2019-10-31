# device capability handling

# select the highest capability that is supported by both the toolchain and device
function supported_capability(dev::CuDevice)
    dev_cap = capability(dev)
    compat_caps = filter(cap -> cap <= dev_cap, target_support[])
    isempty(compat_caps) &&
        error("Device capability v$dev_cap not supported by available toolchain")

    return maximum(compat_caps)
end

# return the capability of the current context's device, or a sane fall-back
function current_capability()
    if initialized[]
        return supported_capability(device())
    else
        # newer devices tend to support cleaner code (higher-level instructions, etc)
        # so target the most recent device as supported by this toolchain
        return maximum(target_support[])
    end
end
