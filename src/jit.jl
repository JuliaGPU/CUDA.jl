
# JIT compilation of Julia code to PTX

export cufunction


#
# main code generation functions
#

function module_setup(mod::LLVM.Module)
    # NOTE: NVPTX::TargetMachine's data layout doesn't match the NVPTX user guide,
    #       so we specify it ourselves
    if Int === Int64
        triple!(mod, "nvptx64-nvidia-cuda")
        datalayout!(mod, "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64")
    else
        triple!(mod, "nvptx-nvidia-cuda")
        datalayout!(mod, "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64")
    end

    # debug info metadata
    push!(metadata(mod), "llvm.module.flags",
          MDNode([ConstantInt(Int32(1)),    # llvm::Module::Error
                  MDString("Debug Info Version"), ConstantInt(DEBUG_METADATA_VERSION())]))
end

# make function names safe for ptxas
sanitize_fn(fn::String) = replace(fn, r"[^aA-zZ0-9_]", "_")

function raise_exception(insblock::BasicBlock, ex::Value)
    fun = LLVM.parent(insblock)
    mod = LLVM.parent(fun)
    ctx = context(mod)

    builder = Builder(ctx)
    position!(builder, insblock)

    trap = if haskey(functions(mod), "llvm.trap")
        get(functions(mod), "llvm.trap")
    else
        LLVM.Function(mod, "llvm.trap", LLVM.FunctionType(LLVM.VoidType(ctx)))
    end
    call!(builder, trap)
end

function irgen(func::ANY, tt::ANY; kernel::Bool=false)
    # sanity checks
    Base.JLOptions().can_inline == 0 &&
        Base.warn_once("inlining disabled, CUDA code generation will almost certainly fail")

    fn = String(typeof(func).name.mt.name)
    mod = LLVM.Module(sanitize_fn(fn))
    module_setup(mod)

    # collect all modules of IR
    # TODO: emit into module instead of parsing
    # TODO: make codegen pure
    hook_module_setup(ref::Ptr{Void}) =
        module_setup(LLVM.Module(convert(LLVM.API.LLVMModuleRef, ref)))
    hook_raise_exception(insblock::Ptr{Void}, ex::Ptr{Void}) =
        raise_exception(BasicBlock(convert(LLVM.API.LLVMValueRef, insblock)),
                        Value(convert(LLVM.API.LLVMValueRef, ex)))
    modrefs = Vector{Ptr{Void}}()
    hook_module_activation(ref::Ptr{Void}) = push!(modrefs, ref)
    hooks = Base.CodegenHooks(module_setup=hook_module_setup,
                              module_activation=hook_module_activation,
                              raise_exception=hook_raise_exception)
    params = Base.CodegenParams(cached=false,
                                runtime=false, exceptions=false,
                                track_allocations=false, code_coverage=false,
                                static_alloc=false, dynamic_alloc=false,
                                hooks=hooks)
    entry_irmod = Base._dump_function(func, tt,
                                      #=native=#false, #=wrapper=#false, #=strip=#false,
                                      #=dump_module=#true, #=syntax=#:att, #=optimize=#false,
                                      params)
    irmods = map(ref->convert(String, LLVM.Module(ref)), modrefs)
    unshift!(irmods, entry_irmod)

    # find all Julia functions
    # TODO: let Julia report this
    julia_fns = Dict{String,Dict{String,String}}()
    let entry_mod = parse(LLVM.Module, entry_irmod)
        r = r"^(?P<cc>(jl|japi|jsys|julia)[^\W_]*)_(?P<name>.+)_\d+$"
        for ir_f in functions(entry_mod)
            ir_fn = LLVM.name(ir_f)
            m = match(r, ir_fn)
            if m != nothing
                fns = get!(julia_fns, m[:name], Dict{String,String}())
                fns[m[:cc]] = ir_fn
            end
        end
    end

    # find the native entry-point function
    haskey(julia_fns, fn) || error("could not find compiled function for $fn")
    entry_fns = julia_fns[fn]
    if !haskey(entry_fns, "julia")
        error("could not find native function for $fn, available CCs are: ",
              join(keys(entry_fns), ", "))
    end
    entry_fn = entry_fns["julia"]

    # link all the modules
    for irmod in irmods
        partial_mod = parse(LLVM.Module, irmod)

        name!(partial_mod, "Julia IR")
        triple!(partial_mod, triple(mod))
        datalayout!(partial_mod, datalayout(mod))

        link!(mod, partial_mod)
    end

    # clean up incompatibilities
    for f in functions(mod)
        if startswith(LLVM.name(f), "jlcall_")
            # we don't need the generic wrapper
            unsafe_delete!(mod, f)
        else
            # only occurs in debug builds
            delete!(function_attributes(f), EnumAttribute("sspreq"))

            # make function names safe for ptxas
            # (LLVM ought to do this, see eg. D17738 and D19126), but fails
            # TODO: fix all globals?
            if !isdeclaration(f)
                orig_fn = LLVM.name(f)
                safe_fn = sanitize_fn(orig_fn)
                if orig_fn != safe_fn
                    LLVM.name!(f, safe_fn)

                    # take care if we're renaming the entry-point function
                    if orig_fn == entry_fn
                        entry_fn = safe_fn
                    end
                end
            end
        end
    end

    # generate a kernel wrapper to fix & improve argument passing
    entry_f = get(functions(mod), entry_fn)
    if kernel
        entry_ft = eltype(llvmtype(entry_f))
        @assert return_type(entry_ft) == LLVM.VoidType()

        # filter out ghost types, which don't occur in the LLVM function signatures
        julia_types = filter(dt->!isghosttype(dt), tt.parameters)

        # generate the wrapper function type & def
        function wrapper_type(julia_t, codegen_t)
            if isa(codegen_t, LLVM.PointerType) && !(julia_t <: Ptr)
                # we didn't specify a pointer, but codegen passes one anyway.
                # make the wrapper accept the underlying value instead.
                return eltype(codegen_t)
            else
                return codegen_t
            end
        end
        wrapper_types = LLVM.LLVMType[wrapper_type(julia_t, codegen_t)
                                      for (julia_t, codegen_t)
                                      in zip(julia_types, parameters(entry_ft))]
        wrapper_fn = "ptxcall" * entry_fn[6:end]
        wrapper_ft = LLVM.FunctionType(LLVM.VoidType(), wrapper_types)
        wrapper_f = LLVM.Function(mod, wrapper_fn, wrapper_ft)

        # emit IR performing the "conversions"
        Builder() do builder
            entry = BasicBlock(wrapper_f, "entry")
            position!(builder, entry)

            wrapper_args = Vector{LLVM.Value}()

            # perform argument conversions
            codegen_types = parameters(entry_ft)
            wrapper_params = parameters(wrapper_f)
            for (julia_t, codegen_t, wrapper_t, wrapper_param) in
                zip(julia_types, codegen_types, wrapper_types, wrapper_params)
                if codegen_t != wrapper_t
                    # the wrapper argument doesn't match the kernel parameter type.
                    # this only happens when codegen wants to pass a pointer.
                    @assert isa(codegen_t, LLVM.PointerType)
                    # copy the argument value to a stack slot, and reference it.
                    ptr = alloca!(builder, wrapper_t)
                    store!(builder, wrapper_param, ptr)
                    ptr_compat = bitcast!(builder, ptr, codegen_t)
                    push!(wrapper_args, ptr_compat)
                else
                    push!(wrapper_args, wrapper_param)
                end
            end

            call!(builder, entry_f, wrapper_args)

            ret!(builder)
        end

        push!(function_attributes(entry_f), EnumAttribute("alwaysinline"))
        linkage!(entry_f, LLVM.API.LLVMInternalLinkage)
        entry_f = wrapper_f
    end
        
    verify(mod)

    return mod, entry_f
end

const libdevices = Dict{VersionNumber, LLVM.Module}()
function link_libdevice!(mod::LLVM.Module, cap::VersionNumber)
    CUDAnative.configured || error("CUDAnative.jl has not been configured; cannot JIT code.")

    # select the most recent & compatible libdevice
    const vers = keys(CUDAnative.libdevice_libraries)
    compat_vers = Set(ver for ver in vers if ver <= cap)
    isempty(compat_vers) && error("No compatible CUDA device library available")
    ver = maximum(compat_vers)
    path = libdevice_libraries[ver]

    # load the library, once
    if !haskey(libdevices, ver)
        open(path) do io
            libdevice_mod = parse(LLVM.Module, read(io))
            name!(libdevice_mod, "libdevice")
            libdevices[ver] = libdevice_mod
        end
    end
    libdevice_mod = LLVM.Module(libdevices[ver])

    # override libdevice's triple and datalayout to avoid warnings
    triple!(libdevice_mod, triple(mod))
    datalayout!(libdevice_mod, datalayout(mod))

    # 1. Save list of external functions
    exports = map(LLVM.name, functions(mod))
    filter!(fn->!haskey(functions(libdevice_mod), fn), exports)

    # 2. Link with libdevice
    link!(mod, libdevice_mod)

    ModulePassManager() do pm
        # 3. Internalize all functions not in list from (1)
        internalize!(pm, exports)

        # 4. Eliminate all unused internal functions
        global_optimizer!(pm)
        global_dce!(pm)
        strip_dead_prototypes!(pm)

        # 5. Run NVVMReflect pass
        push!(metadata(mod), "nvvm-reflect-ftz",
              MDNode([ConstantInt(Int32(1))]))

        # 6. Run standard optimization pipeline
        #
        #    see `optimize!`

        run!(pm, mod)
    end
end

function machine(cap::VersionNumber, triple::String)
    InitializeNVPTXTarget()
    InitializeNVPTXTargetInfo()
    t = Target(triple)

    InitializeNVPTXTargetMC()
    cpu = "sm_$(cap.major)$(cap.minor)"
    tm = TargetMachine(t, triple, cpu)

    return tm
end

function optimize!(mod::LLVM.Module, cap::VersionNumber)
    tm = machine(cap, triple(mod))

    ModulePassManager() do pm
        ccall(:LLVMAddLowerGCFramePass, Void,
              (LLVM.API.LLVMPassManagerRef,), LLVM.ref(pm))
        populate!(pm, tm)
        ccall(:LLVMAddLowerPTLSPass, Void,
              (LLVM.API.LLVMPassManagerRef, Cint), LLVM.ref(pm), 0)

        PassManagerBuilder() do pmb
            always_inliner!(pm) # TODO: set it as the builder's inliner
            populate!(pm, pmb)
        end

        run!(pm, mod)
    end
end

function mcgen(mod::LLVM.Module, func::LLVM.Function, cap::VersionNumber;
               kernel::Bool=true)
    tm = machine(cap, triple(mod))

    # kernel metadata
    if kernel
        push!(metadata(mod), "nvvm.annotations",
             MDNode([func, MDString("kernel"), ConstantInt(Int32(1))]))
    end

    InitializeNVPTXAsmPrinter()
    return convert(String, emit(tm, mod, LLVM.API.LLVMAssemblyFile))
end

# Compile a function to PTX, returning the assembly and an entry point.
# Not to be used directly, see `cufunction` instead.
#
# The `kernel` argument indicates whether we are compiling a kernel entry-point function,
# in which case extra metadata needs to be attached.
function compile_function(func::ANY, tt::ANY, cap::VersionNumber; kernel::Bool=true)
    check_invocation(func, tt; kernel=kernel)

    sig = "$(typeof(func).name.mt.name)($(join(tt.parameters, ", ")))"
    debug("(Re)compiling kernel $sig for device capability $cap")

    # generate LLVM IR
    mod, entry = irgen(func, tt; kernel=kernel)
    trace("Module entry point: ", LLVM.name(entry))

    # link libdevice, if it might be necessary
    if any(f->isdeclaration(f) && intrinsic_id(f)==0, functions(mod))
        link_libdevice!(mod, cap)
    end

    # generate (PTX) assembly
    optimize!(mod, cap)
    module_asm = mcgen(mod, entry, cap; kernel=kernel)

    return module_asm, LLVM.name(entry)
end

# check validity of a function invocation, specified by the generic function and a tupletype
function check_invocation(func::ANY, tt::ANY; kernel::Bool=false)
    sig = "$(typeof(func).name.mt.name)($(join(tt.parameters, ", ")))"

    # get the method
    ms = Base.methods(func, tt)
    isempty(ms)   && throw(ArgumentError("no method found for $sig"))
    length(ms)!=1 && throw(ArgumentError("no unique matching method for $sig"))
    m = first(ms)

    # emulate some of the specsig logic from codegen.cppto detect non-native CC functions
    # TODO: also do this for device functions (#87)
    isleaftype(tt) || throw(ArgumentError("invalid call to device function $sig: passing abstract arguments"))
    m.isva && throw(ArgumentError("invalid device function $sig: is a varargs function"))

    # kernels can't return values
    if kernel
        rt = Base.return_types(func, tt)[1]
        if rt != Void
            throw(ArgumentError("$sig is not a valid kernel as it returns $rt"))
        end
    end
end

# Main entry point for compiling a Julia function + argtypes to a callable CUDA function
function cufunction(dev::CuDevice, func::ANY, types::ANY)
    CUDAnative.configured || error("CUDAnative.jl has not been configured; cannot JIT code.")
    @assert isa(func, Core.Function)
    tt = Base.to_tuple_type(types)

    # select a capability level
    dev_cap = capability(dev)
    compat_caps = filter(cap -> cap <= dev_cap, supported_capabilities)
    isempty(compat_caps) &&
        error("Device capability v$dev_cap not supported by available toolchain")
    cap = maximum(compat_caps)

    (module_asm, module_entry) = compile_function(func, tt, cap)

    # enable debug options based on Julia's debug setting
    jit_options = Dict{CUDAdrv.CUjit_option,Any}()
    if DEBUG || Base.JLOptions().debug_level >= 1
        jit_options[CUDAdrv.GENERATE_LINE_INFO] = true
    end
    if DEBUG || Base.JLOptions().debug_level >= 2
        # TODO: detect cuda-gdb
        jit_options[CUDAdrv.GENERATE_DEBUG_INFO] = true
    end
    cuda_mod = CuModule(module_asm, jit_options)
    cuda_fun = CuFunction(cuda_mod, module_entry)

    return cuda_fun, cuda_mod
end

function init_jit()
    llvm_args = [
        # Program name; can be left blank.
        "",
        # Enable generation of FMA instructions to mimic behavior of nvcc.
        "--nvptx-fma-level=1",
    ]
    LLVM.API.LLVMParseCommandLineOptions(Int32(length(llvm_args)),
        [Base.unsafe_convert(Cstring, llvm_arg) for llvm_arg in llvm_args], C_NULL)
end
