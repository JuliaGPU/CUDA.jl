"""
    @sync [blocking=false] ex

Run expression `ex` and synchronize the GPU afterwards.

The `blocking` keyword argument determines how synchronization is performed. By default,
non-blocking synchronization will be used, which gives other Julia tasks a chance to run
while waiting for the GPU to finish. This may increase latency, so for short operations,
or when benchmaring code that does not use multiple tasks, it may be beneficial to use
blocking synchronization instead by setting `blocking=true`. Blocking synchronization
can also be enabled globally by changing the `nonblocking_synchronization` preference.

See also: [`synchronize`](@ref).
"""
macro sync(ex...)
    # destructure the `@sync` expression
    code = ex[end]
    kwargs = ex[1:end-1]

    # decode keyword arguments
    blocking = false
    for kwarg in kwargs
        Meta.isexpr(kwarg, :(=)) || error("Invalid keyword argument $kwarg")
        key, val = kwarg.args
        if key == :blocking
            isa(val, Bool) ||
                error("Invalid value for keyword argument $kwarg; expected Bool, got $(val)")
            blocking = val
        else
            error("Unknown keyword argument $kwarg")
        end
    end

    quote
        local ret = $(esc(code))
        synchronize(; blocking=$blocking)
        ret
    end
end

@public @sync


"""
    explain_unsupported_device(io::IO)

Print an explanation for a vendor-library error that typically means the library contains
no device code for the current GPU. Prints nothing when there is no device to talk about.
"""
function explain_unsupported_device(io::IO)
    cap = try
        functional() ? capability(device()) : nothing
    catch
        nothing
    end
    cap === nothing && return

    print(io, "\n\nThis can mean the library contains no device code for your GPU " *
              "(compute capability $(cap.major).$(cap.minor)).")
    print(io, " Vendor libraries drop older architectures over time, and NVIDIA builds " *
              "the Tegra redistributables for the JetPack generation they belong to, so " *
              "a library can load and initialize fine and only fail once it launches a " *
              "kernel.")
    if is_tegra()
        print(io, " Refer to the platform support matrix in the CUDA.jl README, and to " *
                  "the Jetson section of the installation guide.")
    else
        print(io, " Refer to the platform support matrix in the CUDA.jl README.")
    end
    return
end
