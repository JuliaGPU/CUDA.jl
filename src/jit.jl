# JIT compilation of Julia code to PTX

export cufunction

struct CompilerContext
    # core invocation
    f::Core.Function
    tt::DataType
    cap::VersionNumber
    kernel::Bool

    # optional properties
    alias::Union{Nothing,String}
    minthreads::Union{Nothing,CuDim}
    maxthreads::Union{Nothing,CuDim}
    blocks_per_sm::Union{Nothing,Integer}
    maxregs::Union{Nothing,Integer}

    # hacks
    inner_f::Union{Nothing,Core.Function}

    CompilerContext(f, tt, cap, kernel; inner_f=nothing, alias=nothing,
                    minthreads=nothing, maxthreads=nothing, blocks_per_sm=nothing, maxregs=nothing) =
        new(f, tt, cap, kernel, alias, minthreads, maxthreads, blocks_per_sm, maxregs, inner_f)
end

struct CompilerError <: Exception
    ctx::CompilerContext
    message::String
    meta::Dict
end

compiler_error(ctx::CompilerContext, message="unknown error"; kwargs...) =
    throw(CompilerError(ctx, message, kwargs))

function Base.showerror(io::IO, err::CompilerError)
    ctx = err.ctx
    fn = typeof(coalesce(ctx.inner_f, ctx.f)).name.mt.name
    args = join(ctx.tt.parameters, ", ")
    print(io, "could not compile $fn($args) for GPU; $(err.message)")
    if haskey(err.meta, :errors) && isa(err.meta[:errors], Vector{UnsupportedIRError})
        for suberr in err.meta[:errors]
            print(io, "\n- ")
            Base.showerror(io, suberr)
        end
        print(io, "\nTry inspecting generated code with the @device_code_... macros")
    else
        for (key,val) in err.meta
            print(io, "\n- $key = $val")
        end
    end
end


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

    # add debug info metadata
    push!(metadata(mod), "llvm.module.flags",
         MDNode([ConstantInt(Int32(1)),    # llvm::Module::Error
                 MDString("Debug Info Version"),
                 ConstantInt(DEBUG_METADATA_VERSION())]))
end

# make function names safe for PTX
safe_fn(fn::String) = replace(fn, r"[^aA-zZ0-9_]"=>"_")
safe_fn(f::Core.Function) = safe_fn(String(typeof(f).name.mt.name))
safe_fn(f::LLVM.Function) = safe_fn(LLVM.name(f))

function raise_exception(insblock::BasicBlock, ex::Value)
    fun = LLVM.parent(insblock)
    mod = LLVM.parent(fun)
    ctx = context(mod)

    builder = Builder(ctx)
    position!(builder, insblock)

    trap = if haskey(functions(mod), "llvm.trap")
        functions(mod)["llvm.trap"]
    else
        LLVM.Function(mod, "llvm.trap", LLVM.FunctionType(LLVM.VoidType(ctx)))
    end
    call!(builder, trap)
end

# maintain our own "global unique" suffix for disambiguating kernels
globalUnique = 0

function irgen(ctx::CompilerContext)
    # get the method instance
    isa(ctx.f, Core.Builtin) && compiler_error(ctx, "function is not a generic function")
    world = typemax(UInt)
    meth = which(ctx.f, ctx.tt)
    sig_tt = Tuple{typeof(ctx.f), ctx.tt.parameters...}
    (ti, env) = ccall(:jl_type_intersection_with_env, Any,
                      (Any, Any), sig_tt, meth.sig)::Core.SimpleVector
    meth = Base.func_for_method_checked(meth, ti)
    linfo = ccall(:jl_specializations_get_linfo, Ref{Core.MethodInstance},
                  (Any, Any, Any, UInt), meth, ti, env, world)

    # set-up the compiler interface
    function hook_module_setup(ref::Ptr{Cvoid})
        ref = convert(LLVM.API.LLVMModuleRef, ref)
        module_setup(LLVM.Module(ref))
    end
    function hook_raise_exception(insblock::Ptr{Cvoid}, ex::Ptr{Cvoid})
        insblock = convert(LLVM.API.LLVMValueRef, insblock)
        ex = convert(LLVM.API.LLVMValueRef, ex)
        raise_exception(BasicBlock(insblock), Value(ex))
    end
    dependencies = Vector{LLVM.Module}()
    function hook_module_activation(ref::Ptr{Cvoid})
        ref = convert(LLVM.API.LLVMModuleRef, ref)
        push!(dependencies, LLVM.Module(ref))
    end
    params = Base.CodegenParams(cached=false,
                                track_allocations=false,
                                code_coverage=false,
                                static_alloc=false,
                                prefer_specsig=true,
                                module_setup=hook_module_setup,
                                module_activation=hook_module_activation,
                                raise_exception=hook_raise_exception)

    # get the code
    mod = let
        ref = ccall(:jl_get_llvmf_defn, LLVM.API.LLVMValueRef,
                    (Any, UInt, Bool, Bool, Base.CodegenParams),
                    linfo, world, #=wrapper=#false, #=optimize=#false, params)
        if ref == C_NULL
            compiler_error(ctx, "the Julia compiler could not generate LLVM IR")
        end

        llvmf = LLVM.Function(ref)
        LLVM.parent(llvmf)
    end

    # the main module should contain a single jfptr_ function definition,
    # e.g. jlcall_kernel_vadd_62977
    definitions = filter(f->!isdeclaration(f), functions(mod))
    wrapper = let
        fs = collect(filter(f->startswith(LLVM.name(f), "jfptr_"), definitions))
        @assert length(fs) == 1
        fs[1]
    end

    # the jlcall wrapper function should point us to the actual entry-point,
    # e.g. julia_kernel_vadd_62984
    entry_tag = let
        m = match(r"jfptr_(.+)_\d+", LLVM.name(wrapper))
        @assert m != nothing
        m.captures[1]
    end
    unsafe_delete!(mod, wrapper)
    entry = let
        re = Regex("julia_$(entry_tag)_\\d+")
        llvmcall_re = Regex("julia_$(entry_tag)_\\d+u\\d+")
        fs = collect(filter(f->occursin(re, LLVM.name(f)) &&
                               !occursin(llvmcall_re, LLVM.name(f)), definitions))
        if length(fs) != 1
            compiler_error(f, tt, cap, "could not find single entry-point";
                           entry=>entry_tag, available=>[LLVM.name.(definitions)])
        end
        fs[1]
    end

    # link in dependent modules
    link!.(Ref(mod), dependencies)

    # clean up incompatibilities
    for llvmf in functions(mod)
        # only occurs in debug builds
        delete!(function_attributes(llvmf), EnumAttribute("sspreq", 0, jlctx[]))

        # make function names safe for ptxas
        # (LLVM ought to do this, see eg. D17738 and D19126), but fails
        # TODO: fix all globals?
        llvmfn = LLVM.name(llvmf)
        if !isdeclaration(llvmf)
            llvmfn′ = safe_fn(llvmf)
            if llvmfn != llvmfn′
                LLVM.name!(llvmf, llvmfn′)
            end
        end
    end

    # rename the entry point
    llvmfn = replace(LLVM.name(entry), r"_\d+$"=>"")
    ## add a friendlier alias
    alias = coalesce(ctx.alias, String(typeof(coalesce(ctx.inner_f, ctx.f)).name.mt.name))
    if startswith(alias, '#')
        alias = "anonymous"
    else
        alias = safe_fn(alias)
    end
    llvmfn = replace(llvmfn, r"_.+" => "_$alias")
    ## append a global unique counter
    global globalUnique
    globalUnique += 1
    llvmfn *= "_$globalUnique"
    LLVM.name!(entry, llvmfn)

    return mod, entry
end

# promote a function to a kernel
# FIXME: sig vs tt (code_llvm vs cufunction)
function promote_kernel!(ctx::CompilerContext, mod::LLVM.Module, entry_f::LLVM.Function)
    kernel = wrap_entry!(ctx, mod, entry_f)


    # property annotations
    # TODO: belongs in irgen? doesn't maxntidx doesn't appear in ptx code?

    annotations = LLVM.Value[kernel]

    ## kernel metadata
    append!(annotations, [MDString("kernel"), ConstantInt(Int32(1))])

    ## expected CTA sizes
    for (typ,vals) in (:req=>ctx.minthreads, :max=>ctx.maxthreads)
        if vals != nothing
            bounds = CUDAdrv.CuDim3(vals)
            for dim in (:x, :y, :z)
                bound = getfield(bounds, dim)
                append!(annotations, [MDString("$(typ)ntid$(dim)"),
                                      ConstantInt(Int32(bound))])
            end
        end
    end

    if ctx.blocks_per_sm != nothing
        append!(annotations, [MDString("minctasm"), ConstantInt(Int32(ctx.blocks_per_sm))])
    end

    if ctx.maxregs != nothing
        append!(annotations, [MDString("maxnreg"), ConstantInt(Int32(ctx.maxregs))])
    end


    push!(metadata(mod), "nvvm.annotations", MDNode(annotations))


    return kernel
end

# generate a kernel wrapper to fix & improve argument passing
function wrap_entry!(ctx::CompilerContext, mod::LLVM.Module, entry_f::LLVM.Function)
    entry_ft = eltype(llvmtype(entry_f))
    @assert return_type(entry_ft) == LLVM.VoidType(jlctx[])

    # filter out ghost types, which don't occur in the LLVM function signatures
    sig = Base.signature_type(ctx.f, ctx.tt)
    julia_types = filter(dt->!isghosttype(dt), sig.parameters)

    # generate the wrapper function type & definition
    global globalUnique
    function wrapper_type(julia_t, codegen_t)
        if !isbitstype(julia_t)
            # don't pass jl_value_t by value; it's an opaque structure
            return codegen_t
        elseif isa(codegen_t, LLVM.PointerType) && !(julia_t <: Ptr)
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
    wrapper_fn = replace(LLVM.name(entry_f), r"^.+?_"=>"ptxcall_") # change the CC tag
    wrapper_ft = LLVM.FunctionType(LLVM.VoidType(jlctx[]), wrapper_types)
    wrapper_f = LLVM.Function(mod, wrapper_fn, wrapper_ft)

    # emit IR performing the "conversions"
    Builder(jlctx[]) do builder
        entry = BasicBlock(wrapper_f, "entry", jlctx[])
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
                @assert eltype(codegen_t) == wrapper_t

                # copy the argument value to a stack slot, and reference it.
                ptr = alloca!(builder, wrapper_t)
                if LLVM.addrspace(codegen_t) != 0
                    ptr = addrspacecast!(builder, ptr, codegen_t)
                end
                store!(builder, wrapper_param, ptr)
                push!(wrapper_args, ptr)

                # Julia marks parameters as TBAA immutable;
                # this is incompatible with us storing to a stack slot, so clear TBAA
                # TODO: tag with alternative information (eg. TBAA, or invariant groups)
                entry_params = collect(parameters(entry_f))
                candidate_uses = []
                for param in entry_params
                    append!(candidate_uses, collect(uses(param)))
                end
                while !isempty(candidate_uses)
                    usepair = popfirst!(candidate_uses)
                    inst = user(usepair)

                    md = metadata(inst)
                    if haskey(md, LLVM.MD_tbaa)
                        delete!(md, LLVM.MD_tbaa)
                    end

                    # follow along certain pointer operations
                    if isa(inst, LLVM.GetElementPtrInst) ||
                       isa(inst, LLVM.BitCastInst) ||
                       isa(inst, LLVM.AddrSpaceCastInst)
                        append!(candidate_uses, collect(uses(inst)))
                    end
                end
            else
                push!(wrapper_args, wrapper_param)
            end
        end

        call!(builder, entry_f, wrapper_args)

        ret!(builder)
    end

    # early-inline the original entry function into the wrapper
    push!(function_attributes(entry_f), EnumAttribute("alwaysinline", 0, jlctx[]))
    linkage!(entry_f, LLVM.API.LLVMInternalLinkage)
    ModulePassManager() do pm
        always_inliner!(pm)
        run!(pm, mod)
    end

    return wrapper_f
end

const libdevices = Dict{String, LLVM.Module}()
function link_libdevice!(ctx::CompilerContext, mod::LLVM.Module)
    CUDAnative.configured || return

    # find libdevice
    path = if isa(libdevice, Dict)
        # select the most recent & compatible library
        vers = keys(CUDAnative.libdevice)
        compat_vers = Set(ver for ver in vers if ver <= ctx.cap)
        isempty(compat_vers) && error("No compatible CUDA device library available")
        ver = maximum(compat_vers)
        path = libdevice[ver]
    else
        libdevice
    end

    # load the library, once
    if !haskey(libdevices, path)
        open(path) do io
            libdevice_mod = parse(LLVM.Module, read(io), jlctx[])
            name!(libdevice_mod, "libdevice")
            libdevices[path] = libdevice_mod
        end
    end
    libdevice_mod = LLVM.Module(libdevices[path])

    # override libdevice's triple and datalayout to avoid warnings
    triple!(libdevice_mod, triple(mod))
    datalayout!(libdevice_mod, datalayout(mod))

    # 1. save list of external functions
    exports = map(LLVM.name, functions(mod))
    filter!(fn->!haskey(functions(libdevice_mod), fn), exports)

    # 2. link with libdevice
    link!(mod, libdevice_mod)

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
              MDNode([ConstantInt(Int32(1))]))

        # 6. run standard optimization pipeline
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
    if cuda_driver_version >= v"9.0" && v"6.0" in ptx_support
        # in the case of CUDA 9, we use sync intrinsics from PTX ISA 6.0+
        feat = "+ptx60"
    else
        feat = ""
    end
    tm = TargetMachine(t, triple, cpu, feat)

    return tm
end

# Optimize a bitcode module according to a certain device capability.
function optimize!(ctx::CompilerContext, mod::LLVM.Module, entry::LLVM.Function)
    tm = machine(ctx.cap, triple(mod))

    # GPU code is _very_ sensitive to register pressure and local memory usage,
    # so forcibly inline every function definition into the entry point
    # and internalize all other functions (enabling ABI-breaking optimizations).
    # FIXME: this is too coarse. use a proper inliner tuned for GPUs
    ModulePassManager() do pm
        no_inline = EnumAttribute("noinline", 0, jlctx[])
        always_inline = EnumAttribute("alwaysinline", 0, jlctx[])
        for f in filter(f->f!=entry && !isdeclaration(f), functions(mod))
            attrs = function_attributes(f)
            if !(no_inline in collect(attrs))
                push!(attrs, always_inline)
            end
            linkage!(f, LLVM.API.LLVMInternalLinkage)
        end
        always_inliner!(pm)
        run!(pm, mod)
    end

    ModulePassManager() do pm
        add_library_info!(pm, triple(mod))
        add_transform_info!(pm, tm)
        ccall(:jl_add_optimization_passes, Cvoid,
              (LLVM.API.LLVMPassManagerRef, Cint),
              LLVM.ref(pm), Base.JLOptions().opt_level)

        # CUDAnative's JIT internalizes non-inlined child functions, making it possible
        # to rewrite them (whereas the Julia JIT caches those functions exactly);
        # this opens up some more optimization opportunities
        dead_arg_elimination!(pm)   # parent doesn't use return value --> ret void

        global_optimizer!(pm)
        global_dce!(pm)
        strip_dead_prototypes!(pm)

        run!(pm, mod)
    end
end

function mcgen(ctx::CompilerContext, mod::LLVM.Module, f::LLVM.Function)
    tm = machine(ctx.cap, triple(mod))

    InitializeNVPTXAsmPrinter()
    return String(emit(tm, mod, LLVM.API.LLVMAssemblyFile))
end

# Compile a function to PTX, returning the assembly and an entry point.
# Not to be used directly, see `cufunction` instead.
function compile_function(ctx::CompilerContext)
    ## high-level code generation (Julia AST)

    @debug "(Re)compiling function" ctx

    validate_invocation(ctx)


    ## low-level code generation (LLVM IR)

    mod, entry = irgen(ctx)

    if ctx.kernel
        entry = promote_kernel!(ctx, mod, entry)
    end

    @trace("Module entry point: ", LLVM.name(entry))

    # link libdevice, if it might be necessary
    # TODO: should be more find-grained -- only matching functions actually in this libdevice
    if any(f->isdeclaration(f) && intrinsic_id(f)==0, functions(mod))
        link_libdevice!(ctx, mod)
    end

    # optimize the IR (otherwise the IR isn't necessarily compatible)
    optimize!(ctx, mod, entry)

    # make sure any non-isbits arguments are unused
    real_arg_i = 0
    sig = Base.signature_type(ctx.f, ctx.tt)
    for (arg_i,dt) in enumerate(sig.parameters)
        isghosttype(dt) && continue
        real_arg_i += 1

        if !isbitstype(dt)
            param = parameters(entry)[real_arg_i]
            if !isempty(uses(param))
                compiler_error(ctx, "passing and using non-bitstype argument";
                               argument=arg_i, argument_type=dt)
            end
        end
    end

    # validate generated IR
    errors = validate_ir(mod)
    if !isempty(errors)
        compiler_error(ctx, "unsupported LLVM IR"; errors=errors)
    end


    ## machine code generation (PTX assembly)

    module_asm = mcgen(ctx, mod, entry)

    return module_asm, LLVM.name(entry)
end

# (::CompilerContext)
const compile_hook = Ref{Union{Nothing,Function}}(nothing)

# Main entry point for compiling a Julia function + argtypes to a callable CUDA function
function cufunction(dev::CuDevice, @nospecialize(f), @nospecialize(tt); kwargs...)
    CUDAnative.configured || error("CUDAnative.jl has not been configured; cannot JIT code.")
    @assert isa(f, Core.Function)

    # select a capability level
    dev_cap = capability(dev)
    compat_caps = filter(cap -> cap <= dev_cap, target_support)
    isempty(compat_caps) &&
        error("Device capability v$dev_cap not supported by available toolchain")
    cap = maximum(compat_caps)

    ctx = CompilerContext(f, tt, cap, true; kwargs...)

    if compile_hook[] != nothing
        global globalUnique
        previous_globalUnique = globalUnique
        compile_hook[](ctx)
        globalUnique = previous_globalUnique
    end

    (module_asm, module_entry) = compile_function(ctx)

    # enable debug options based on Julia's debug setting
    jit_options = Dict{CUDAdrv.CUjit_option,Any}()
    if Base.JLOptions().debug_level == 1
        jit_options[CUDAdrv.GENERATE_LINE_INFO] = true
    elseif Base.JLOptions().debug_level >= 2
        jit_options[CUDAdrv.GENERATE_DEBUG_INFO] = true
    end
    cuda_mod = CuModule(module_asm, jit_options)
    cuda_fun = CuFunction(cuda_mod, module_entry)

    @debug begin
        attr = attributes(cuda_fun)
        bin_ver = VersionNumber(divrem(attr[CUDAdrv.FUNC_ATTRIBUTE_BINARY_VERSION],10)...)
        ptx_ver = VersionNumber(divrem(attr[CUDAdrv.FUNC_ATTRIBUTE_PTX_VERSION],10)...)
        regs = attr[CUDAdrv.FUNC_ATTRIBUTE_NUM_REGS]
        local_mem = attr[CUDAdrv.FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES]
        shared_mem = attr[CUDAdrv.FUNC_ATTRIBUTE_SHARED_SIZE_BYTES]
        constant_mem = attr[CUDAdrv.FUNC_ATTRIBUTE_CONST_SIZE_BYTES]
        """Compiled $f to PTX $ptx_ver for SM $bin_ver using $regs registers.
           Memory usage: $local_mem B local, $shared_mem B shared, $constant_mem B constant"""
    end

    return cuda_fun, cuda_mod
end

function init_jit()
    # enable generation of FMA instructions to mimic behavior of nvcc
    LLVM.clopts("--nvptx-fma-level=1")
end
