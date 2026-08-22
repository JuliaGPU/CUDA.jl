using CUDACore: functional, runtime_version, driver_version, compiler_version,
                devices, name, capability, device!, free_memory, total_memory,
                uuid, parent_uuid, CuDevice, CUDA_Runtime_jll, CUDA_Driver_jll

@public versioninfo

function versioninfo(io::IO=stdout)
    @assert functional(true)

    println(io, "CUDA toolchain: ")

    rv = runtime_version()
    print(io, "- runtime $(rv.major).$(rv.minor).$(rv.patch), ")
    if CUDACore.local_toolkit
        println(io, "local installation")
    else
        println(io, "artifact installation")
    end
    if has_nvml()
        print(io, "- driver $(NVML.driver_version())")
    else
        print(io, "- unknown driver")
    end
    println(io, " for $(driver_version().major).$(driver_version().minor)")
    cv = compiler_version()
    print(io, "- compiler $(cv.major).$(cv.minor).$(cv.patch), ")
    if CUDACore.local_compiler
        println(io, "local installation")
    else
        println(io, "artifact installation")
    end
    println(io)

    println(io, "CUDA libraries: ")
    for (name, uuid) in [
        "cuBLAS"      => Base.UUID("182d3088-87b7-4494-8cad-fc6afaa545bc"),
        "cuSPARSE"    => Base.UUID("b26da814-b3bc-49ef-b0ee-c816305aa060"),
        "cuSOLVER"    => Base.UUID("887afef0-6a32-4de5-add4-7827692ba8fc"),
        "cuFFT"       => Base.UUID("533571aa-0936-420e-b4be-9c66f5f626ca"),
        "cuRAND"      => Base.UUID("20fd9a0b-12d5-4c2f-a8af-7c34e9e60431"),
        "cuDNN"       => Base.UUID("02a925ec-e4fe-4b08-9a7e-0d78e3d38ccd"),
        "cuTENSOR"    => Base.UUID("011b41b2-24ef-40a8-b3eb-fa098493e9e1"),
        "cuTensorNet" => Base.UUID("448d79b3-4b49-4e06-a5ea-00c62c0dc3db"),
        "cuStateVec"  => Base.UUID("92f7fd98-d22e-4c0d-85a8-6ade11b672fb"),
    ]
        pkgid = Base.PkgId(uuid, name)
        mod = get(Base.loaded_modules, pkgid, nothing)
        mod === nothing && continue
        if mod.functional()
            println(io, "- $name: $(mod.version())")
        else
            println(io, "- $name: missing")
        end
    end
    println(io, "- CUPTI: $(CUPTI.library_version()) (API $(CUPTI.version()))")
    println(io, "- NVML: ", has_nvml() ? NVML.version() : "missing")
    println(io)

    get_module(name::Symbol) = (name, getfield(CUDACore, name))
    function get_module(pkg::Tuple{String, String})
        id = Base.PkgId(Base.UUID(pkg[1]), pkg[2])
        (pkg[2], get(Base.loaded_modules, id, nothing))
    end

    println(io, "Julia packages: ")
    println(io, "- CUDACore: $(Base.pkgversion(CUDACore))")
    for pkg in [:GPUArrays, :GPUCompiler, ("63c18a36-062a-441e-b654-da1e3ab1ce7c", "KernelAbstractions"),
                 :CUDA_Driver_jll, :CUDA_Compiler_jll, ("76a88914-d11a-5bdc-97e0-2f5a05c973a2", "CUDA_Runtime_jll"),
                 ("1af6417a-86b4-443c-805f-a4643ffb695f", "CUDA_Runtime_Discovery"), :NVPTX_LLVM_Backend_jll]
        name, mod = get_module(pkg)
        isnothing(mod) || println(io, "- $(name): $(Base.pkgversion(mod))")
    end
    println(io)

    println(io, "Toolchain:")
    println(io, "- Julia: $VERSION")
    println(io, "- LLVM: $(LLVM.version())")
    println(io)

    env = filter(var->startswith(var, "JULIA_CUDA"), keys(ENV))
    if !isempty(env)
        println(io, "Environment:")
        for var in env
            println(io, "- $var: $(ENV[var])")
        end
        println(io)
    end

    prefs = [
        "nonblocking_synchronization" => Preferences.load_preference(CUDACore, "nonblocking_synchronization"),
        "default_memory" => Preferences.load_preference(CUDACore, "default_memory"),
        "CUDA_Runtime_jll.version" => Preferences.load_preference(CUDA_Runtime_jll, "version"),
        "CUDA_Runtime_jll.local" => Preferences.load_preference(CUDA_Runtime_jll, "local"),
        "CUDA_Compiler_jll.version" => Preferences.load_preference(CUDA_Compiler_jll, "version"),
        "CUDA_Compiler_jll.local" => Preferences.load_preference(CUDA_Compiler_jll, "local"),
        "CUDA_Driver_jll.compat" => Preferences.load_preference(CUDA_Driver_jll, "compat"),
    ]
    if any(x->!isnothing(x[2]), prefs)
        println(io, "Preferences:")
        for (key, val) in prefs
            if !isnothing(val)
                println(io, "- $key: $val")
            end
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
                (free=free_memory(), total=total_memory())
            end
            (; str, cap, mem)
        end

        str, cap, mem = if has_nvml()
            try
                query_nvml()
            catch err
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

        # report the default compilation target we'd select for this device
        config = try
            CUDACore.compiler_config(dev)
        catch
            nothing
        end
        if config !== nothing
            ptxas_sm  = config.params.sm
            ptxas_ptx = config.params.ptx
            llvm_sm   = CUDACore.SMVersion(config.target.cap.major,
                                           config.target.cap.minor,
                                           config.target.feature_set)
            llvm_ptx  = config.target.ptx
            ptxas_str = "$(CUDACore.cpu_name(ptxas_sm)) / PTX $(ptxas_ptx.major).$(ptxas_ptx.minor)"
            if llvm_sm == ptxas_sm && llvm_ptx == ptxas_ptx
                println(io, "     compiles to $ptxas_str")
            else
                llvm_str = "$(CUDACore.cpu_name(llvm_sm)) / PTX $(llvm_ptx.major).$(llvm_ptx.minor)"
                println(io, "     compiles to $ptxas_str (LLVM: $llvm_str)")
            end
        end
    end
end
