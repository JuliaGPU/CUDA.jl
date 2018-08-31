# machine code generation

## libdevice

function find_libdevice(cap)
    CUDAnative.configured || return
    global libdevice

    if isa(libdevice, Dict)
        # select the most recent & compatible library
        vers = keys(CUDAnative.libdevice)
        compat_vers = Set(ver for ver in vers if ver <= cap)
        isempty(compat_vers) && error("No compatible CUDA device library available")
        ver = maximum(compat_vers)
        path = libdevice[ver]
    else
        libdevice
    end
end

const libdevices = Dict{String, LLVM.Module}()
function load_libdevice(ctx::CompilerContext)
    path = find_libdevice(ctx.cap)

    get!(libdevices, path) do
        open(path) do io
            libdevice = parse(LLVM.Module, read(io), JuliaContext())
            name!(libdevice, "libdevice")
            libdevice
        end
    end
end

function link_libdevice!(ctx::CompilerContext, mod::LLVM.Module, libdevice::LLVM.Module)
    # override libdevice's triple and datalayout to avoid warnings
    triple!(libdevice, triple(mod))
    datalayout!(libdevice, datalayout(mod))

    # 1. save list of external functions
    exports = String[]
    for f in functions(mod)
        fn = LLVM.name(f)
        if !haskey(functions(libdevice), fn)
            push!(exports, fn)
        end
    end

    # 2. link with libdevice
    link!(mod, libdevice)

    ModulePassManager() do pm
        # 3. internalize all functions not in list from (1)
        internalize!(pm, exports)

        # 4. eliminate all unused internal functions
        #
        # this isn't necessary, as we do the same in optimize! to inline kernel wrappers,
        # but it results _much_ smaller modules which are easier to handle on optimize=false
        global_optimizer!(pm)
        global_dce!(pm)
        strip_dead_prototypes!(pm)

        # 5. run NVVMReflect pass
        push!(metadata(mod), "nvvm-reflect-ftz",
              MDNode([ConstantInt(Int32(1), JuliaContext())]))

        # 6. run standard optimization pipeline
        #
        #    see `optimize!`

        run!(pm, mod)
    end
end


## PTX code generation

function machine(cap::VersionNumber, triple::String)
    InitializeNVPTXTarget()
    InitializeNVPTXTargetInfo()
    t = Target(triple)

    InitializeNVPTXTargetMC()
    cpu = "sm_$(cap.major)$(cap.minor)"
    if cuda_driver_version >= v"9.0" && v"6.0" in ptx_support
        # in the case of CUDA 9, we use sync intrinsics from PTX ISA 6.0+
        feat = "+ptx60"
    else
        feat = ""
    end
    tm = TargetMachine(t, triple, cpu, feat)
    asm_verbosity!(tm, true)

    return tm
end

function mcgen(ctx::CompilerContext, mod::LLVM.Module, f::LLVM.Function)
    tm = machine(ctx.cap, triple(mod))

    InitializeNVPTXAsmPrinter()
    return String(emit(tm, mod, LLVM.API.LLVMAssemblyFile))
end
