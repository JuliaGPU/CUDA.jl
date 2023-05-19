"""
    @sync ex

Run expression `ex` and synchronize the GPU afterwards.

See also: [`synchronize`](@ref).
"""
macro sync(ex...)
    # destructure the `@sync` expression
    code = ex[end]
    kwargs = ex[1:end-1]

    # decode keyword arguments
    for kwarg in kwargs
        Meta.isexpr(kwarg, :(=)) || error("Invalid keyword argument $kwarg")
        key, val = kwarg.args
        if key == :blocking
            Base.depwarn("the blocking keyword to @sync has been deprecated", :sync)
        else
            error("Unknown keyword argument $kwarg")
        end
    end

    quote
        local ret = $(esc(code))
        synchronize()
        ret
    end
end

function versioninfo(io::IO=stdout)
    @assert functional(true)

    print(io, "CUDA runtime $(runtime_version().major).$(runtime_version().minor), ")
    if CUDA_Runtime == CUDA_Runtime_jll
        println(io, "artifact installation")
    else
        println(io, "local installation")
    end
    println(io, "CUDA driver $(driver_version().major).$(driver_version().minor)")
    if has_nvml()
        print(io, "NVIDIA driver $(NVML.driver_version())")
    else
        print(io, "Unknown NVIDIA driver")
    end
    if system_driver_version() !== nothing
        println(io, ", originally for CUDA $(system_driver_version().major).$(system_driver_version().minor)")
    else
        println(io)
    end
    println(io)

    println(io, "CUDA libraries: ")
    for lib in (:CUBLAS, :CURAND, :CUFFT, :CUSOLVER, :CUSPARSE)
        mod = getfield(CUDA, lib)
        println(io, "- $lib: ", mod.version())
    end
    println(io, "- CUPTI: ", CUPTI.version())
    println(io, "- NVML: ", has_nvml() ? NVML.version() : "missing")
    println(io)

    println(io, "Julia packages: ")
    ## get a hold of Pkg without adding a dependency on the package
    Pkg = let
        id = Base.PkgId(Base.UUID("44cfe95a-1eb2-52ea-b672-e2afdf69b78f"), "Pkg")
        Base.loaded_modules[id]
    end
    ## look at the Project.toml to determine our version
    project = Pkg.Operations.read_project(Pkg.Types.projectfile_path(pkgdir(CUDA)))
    println(io, "- CUDA.jl: $(project.version)")
    ## dependencies
    deps = Pkg.dependencies()
    versions = Dict(map(uuid->deps[uuid].name => deps[uuid].version, collect(keys(deps))))
    for dep in ["CUDA_Driver_jll", "CUDA_Runtime_jll", "CUDA_Runtime_Discovery"]
        println(io, "- $dep: $(versions[dep])")
    end
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
        function query_nvml()
            mig = uuid(dev) != parent_uuid(dev)
            nvml_gpu = NVML.Device(parent_uuid(dev))
            nvml_dev = NVML.Device(uuid(dev); mig)

            str = NVML.name(nvml_dev)
            cap = NVML.compute_capability(nvml_gpu)
            mem = NVML.memory_info(nvml_dev)

            (; str, cap, mem)
        end

        function query_cuda()
            str = name(dev)
            cap = capability(dev)
            mem = device!(dev) do
                # this requires a device context, so we prefer NVML
                (free=available_memory(), total=total_memory())
            end
            (; str, cap, mem)
        end

        str, cap, mem = if has_nvml()
            try
                query_nvml()
            catch err
                @show err
                if !isa(err, NVML.NVMLError) ||
                   !in(err.code, [NVML.ERROR_NOT_SUPPORTED, NVML.ERROR_NO_PERMISSION])
                    rethrow()
                end
                query_cuda()
            end
        else
            query_cuda()
        end
        println(io, "  $(i-1): $str (sm_$(cap.major)$(cap.minor), $(Base.format_bytes(mem.free)) / $(Base.format_bytes(mem.total)) available)")
    end
end

# this helper function encodes options for compute-sanitizer useful with Julia applications
function compute_sanitizer_cmd(tool::String="memcheck")
    sanitizer = CUDA.compute_sanitizer()
    `$sanitizer --tool $tool --launch-timeout=0 --target-processes=all --report-api-errors=no`
end

"""
    run_compute_sanitizer([julia_args=``]; [tool="memcheck", sanitizer_args=``])

Run a new Julia session under the CUDA compute-sanitizer tool `tool`. This is useful to
detect various GPU-related issues, like memory errors or race conditions.
"""
function run_compute_sanitizer(julia_args=``; tool::String="memcheck", sanitizer_args=``)
    cmd = `$(Base.julia_cmd()) --project=$(Base.active_project())`

    println("Re-starting your active Julia session...")
    run(`$(CUDA.compute_sanitizer_cmd(tool)) $sanitizer_args $cmd $julia_args`)
end
