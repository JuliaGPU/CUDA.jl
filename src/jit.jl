# JIT compilation of Julia code to PTX

export cufunction


#
# code_* replacements
#

# NOTE: default capability is a sane one for testing purposes

function code_llvm(io::IO, f::ANY, types::ANY=Tuple;
                   optimize::Bool=true, dump_module::Bool=false, cap::VersionNumber=v"2.0")
    mod, entry = irgen(f, types)
    if optimize
        optimize!(mod, cap)
    end
    if dump_module
        show(io, mod)
    else
        show(io, entry)
    end
end

code_llvm(f::ANY, types::ANY=Tuple; kwargs...) = code_llvm(STDOUT, f, types; kwargs...)

function code_native(io::IO, f::ANY, types::ANY=Tuple;
                     cap::VersionNumber=v"2.0")
    mod, entry = irgen(f, types)
    optimize!(mod, cap)
    print(io, mcgen(mod, entry, cap))
end

code_native(f::ANY, types::ANY=Tuple; kwargs...) = code_native(STDOUT, f, types; kwargs...)


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

function irgen(func::ANY, tt::ANY)
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

    # find the entry function
    # TODO: let Julia report this
    entry_fn = Nullable{String}()
    let entry_mod = parse(LLVM.Module, entry_irmod)
        for ir_f in functions(entry_mod)
            ir_fn = LLVM.name(ir_f)
            if startswith(ir_fn, "julia_$fn") # FIXME: LLVM might have mangled this
                entry_fn = Nullable(ir_fn)
                break
            end
        end

        isnull(entry_fn) && error("could not find entry-point function")
    end

    # link all the modules
    for irmod in irmods
        partial_mod = parse(LLVM.Module, irmod)

        name!(partial_mod, "Julia IR")
        triple!(partial_mod, triple(mod))
        datalayout!(partial_mod, datalayout(mod))

        link!(mod, partial_mod)
    end

    # FIXME: clean up incompatibilities
    for f in functions(mod)
        if startswith(LLVM.name(f), "jlcall_")
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
                    if orig_fn == get(entry_fn)
                        entry_fn = Nullable(safe_fn)
                    end
                end
            end
        end
    end
        
    verify(mod)

    return mod, get(functions(mod), get(entry_fn))
end

const libdevices = Dict{VersionNumber, LLVM.Module}()
function link_libdevice!(mod::LLVM.Module, cap::VersionNumber)
    # figure out which libdevice versions are compatible with the selected capability
    const vers = [v"2.0", v"3.0", v"3.5"]
    compat_vers = Set(ver for ver in vers if ver <= cap)
    isempty(compat_vers) && error("No compatible CUDA device library available")
    ver = maximum(compat_vers)

    # load the library, once
    if !haskey(libdevices, ver)
        fn = "libdevice.compute_$(ver.major)$(ver.minor).10.bc"

        if haskey(ENV, "NVVMIR_LIBRARY_DIR")
            dirs = [ENV["NVVMIR_LIBRARY_DIR"]]
        else
            dirs = ["/usr/lib/nvidia-cuda-toolkit/libdevice",
                    "/usr/local/cuda/nvvm/libdevice",
                    "/opt/cuda/nvvm/libdevice"]
        end
        any(isdir, dirs) ||
            error("CUDA device library path not found -- specify using NVVMIR_LIBRARY_DIR")

        paths = filter(p->isfile(p), map(d->joinpath(d,fn), dirs))
        isempty(paths) && error("CUDA device library $fn not found")
        path = first(paths)

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
        nvvm_reflect!(pm, Dict("__CUDA_FTZ" => 1))

        # 6. Run standard optimization pipeline
        always_inliner!(pm)

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

function mcgen(mod::LLVM.Module, entry::LLVM.Function, cap::VersionNumber)
    tm = machine(cap, triple(mod))

    # kernel metadata
    push!(metadata(mod), "nvvm.annotations",
          MDNode([entry, MDString("kernel"), ConstantInt(Int32(1))]))

    InitializeNVPTXAsmPrinter()
    return convert(String, emit(tm, mod, LLVM.API.LLVMAssemblyFile))
end

"""
Compile a function to PTX, returning the assembly and an entry point.
Not to be used directly, see `cufunction` instead.
"""
function compile_function(dev::CuDevice, func::ANY, tt::ANY)
    sig = """$func($(join(tt.parameters, ", ")))"""
    debug("Compiling $sig")

    # select a capability level
    dev_cap = capability(dev)
    compat_caps = filter(cap -> cap <= dev_cap, toolchain_caps)
    isempty(compat_caps) &&
        error("Device capability v$dev_cap not supported by available toolchain")
    cap = maximum(compat_caps)
    trace("Targeting $cap architecture")

    @static if TRACE
        # generate a safe and unique name
        function_uid = "$func-"
        if length(tt.parameters) > 0
            function_uid *= join([replace(string(t), r"\W", "") for t in tt.parameters], '.')
        else
            function_uid *= "Void"
        end

        # dump the typed AST
        buf = IOBuffer()
        code_warntype(buf, func, tt)
        ast = String(buf)

        output = "$(dumpdir[])/$function_uid.jl"
        trace("Writing kernel AST to $output")
        open(output, "w") do io
            write(io, ast)
        end
    end

    # Check method validity
    ml = Base.methods(func, tt)
    if length(ml) == 0
        error("no method found for kernel $sig")
    elseif length(ml) > 1
        # TODO: when does this happen?
        error("ambiguous call to kernel $sig")
    end
    rt = Base.return_types(func, tt)[1]
    if rt != Void
        error("cannot call kernel $sig as it returns $rt")
    end

    # generate LLVM IR
    mod, entry = irgen(func, tt)
    @static if TRACE
        output = "$(dumpdir[])/$function_uid.ll"
        trace("Writing kernel LLVM IR to $output")
        open(output, "w") do io
            write(io, convert(String, mod))
        end
    end
    trace("Module entry point: ", LLVM.name(entry))

    # link libdevice, if it might be necessary
    if any(f->isdeclaration(f) && intrinsic_id(f)==0, functions(mod))
        link_libdevice!(mod, cap)
    end

    # generate (PTX) assembly
    optimize!(mod, cap)
    module_asm = mcgen(mod, entry, cap)
    @static if TRACE
        output = "$(dumpdir[])/$function_uid.ptx"
        trace("Writing kernel PTX assembly to $output")
        open(output, "w") do io
            write(io, module_asm)
        end
    end

    return module_asm, LLVM.name(entry)
end

# Main entry-point for compiling a Julia function + argtypes to a callable CUDA function
function cufunction(dev::CuDevice, func::ANY, types::ANY)
    @assert isa(func, Core.Function)
    tt = Base.to_tuple_type(types)

    (module_asm, module_entry) = compile_function(dev, func, tt)

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
