"""
    @sync [blocking=true] ex

Run expression `ex` and synchronize the GPU afterwards. By default, this is a CPU-friendly
synchronization, i.e. it performs a blocking synchronization without increasing CPU load
It is also useful for timing code that executes asynchronously.

See also: [`synchronize`](@ref).
"""
macro sync(ex...)
    # destructure the `@sync` expression
    code = ex[end]
    kwargs = ex[1:end-1]

    # decode keyword arguments
    blocking = true
    for kwarg in kwargs
        Meta.isexpr(kwarg, :(=)) || error("Invalid keyword argument $kwarg")
        key, val = kwarg.args
        if key == :blocking
            blocking = val
        else
            error("Unknown keyword argument $kwarg")
        end
    end

    quote
        local ret = $(esc(code))
        synchronize(; blocking=$(esc(blocking)))
        ret
    end
end

function versioninfo(io::IO=stdout)
    println(io, "CUDA toolkit $(toolkit_version()), $(toolkit_origin()) installation")
    println(io, "CUDA driver $(release())")
    if has_nvml()
        println(io, "NVIDIA driver $(NVML.driver_version())")
    end
    println(io)

    println(io, "Libraries: ")
    for lib in (:CUBLAS, :CURAND, :CUFFT, :CUSOLVER, :CUSPARSE)
        mod = getfield(CUDA, lib)
        println(io, "- $lib: ", mod.version())
    end
    println(io, "- CUPTI: ", has_cupti() ? CUPTI.version() : "missing")
    println(io, "- NVML: ", has_nvml() ? NVML.version() : "missing")
    println(io, "- CUDNN: ", has_cudnn() ? "$(CUDNN.version()) (for CUDA $(CUDNN.cuda_version()))" : "missing")
    println(io, "- CUTENSOR: ", has_cutensor() ? "$(CUTENSOR.version()) (for CUDA $(CUTENSOR.cuda_version()))" : "missing")
    println(io)

    println(io, "Toolchain:")
    println(io, "- Julia: $VERSION")
    println(io, "- LLVM: $(LLVM.version())")
    println(io, "- PTX ISA support: $(join(map(ver->"$(ver.major).$(ver.minor)", supported_toolchain().ptx), ", "))")
    println(io, "- Device capability support: $(join(map(ver->"sm_$(ver.major)$(ver.minor)", supported_toolchain().cap), ", "))")
    println(io)

    env = filter(var->startswith(var, "JULIA_CUDA"), keys(ENV))
    if !isempty(env)
        println(io, "Environment:")
        for var in env
            println(io, "- $var: $(ENV[var])")
        end
        println(io)
    end

    devs = devices()
    if isempty(devs)
        println(io, "No CUDA-capable devices.")
    elseif length(devs) == 1
        println(io, "1 device:")
    else
        println(io, length(devs), " devices:")
    end
    for (i, dev) in enumerate(devs)
        if has_nvml()
            dev′ = NVML.Device(uuid(dev))

            str = NVML.name(dev′)
            cap = NVML.compute_capability(dev′)
            mem = NVML.memory_info(dev′)
        else
            str = name(dev)
            cap = capability(dev)
            mem = device!(dev) do
                # this requires a device context, so we prefer NVML
                (free=available_memory(), total=total_memory())
            end
        end
        println(io, "  $(i-1): $str (sm_$(cap.major)$(cap.minor), $(Base.format_bytes(mem.free)) / $(Base.format_bytes(mem.total)) available)")
    end
end
