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

function signature(ctx::CompilerContext)
    fn = typeof(something(ctx.inner_f, ctx.f)).name.mt.name
    args = join(ctx.tt.parameters, ", ")
    return "$fn($(join(ctx.tt.parameters, ", ")))"
end

abstract type AbstractCompilerError <: Exception end

struct CompilerError <: AbstractCompilerError
    ctx::CompilerContext
    message::String
    bt::StackTraces.StackTrace
    meta::Dict

    CompilerError(ctx::CompilerContext, message="unknown error",
                  bt=StackTraces.StackTrace(); kwargs...) =
        new(ctx, message, bt, kwargs)
end

function Base.showerror(io::IO, err::CompilerError)
    print(io, "CompilerError: could not compile $(signature(err.ctx)); $(err.message)")
    for (key,val) in err.meta
        print(io, "\n- $key = $val")
    end
    Base.show_backtrace(io, err.bt)
end


#
# main code generation functions
#

function module_setup(mod::LLVM.Module)
    triple!(mod, Int === Int64 ? "nvptx64-nvidia-cuda" : "nvptx-nvidia-cuda")

    # add debug info metadata
    push!(metadata(mod), "llvm.module.flags",
         MDNode([ConstantInt(Int32(1), JuliaContext()),    # llvm::Module::Error
                 MDString("Debug Info Version"),
                 ConstantInt(DEBUG_METADATA_VERSION(), JuliaContext())]))
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

# generate a pseudo-backtrace from a stack of methods being emitted
function backtrace(ctx::CompilerContext, method_stack::Vector{Core.MethodInstance})
    bt = StackTraces.StackFrame[]
    for method_instance in method_stack
        # wrapping the kernel doesn't trigger another emit_function,
        # so manually get a hold of the inner function.
        method = if method_instance.def.name == :KernelWrapper
            @assert ctx.inner_f != nothing
            tt = method_instance.specTypes.parameters[2:end]
            which(ctx.inner_f, tt)
        else
            method_instance.def
        end

        frame = StackTraces.StackFrame(method.name, method.file, method.line)
        pushfirst!(bt, frame)
    end
    bt
end

# NOTE: we use an exception to be able to display a stack trace using the logging framework
struct MethodSubstitutionWarning <: Exception
    original::Method
    substitute::Method
end
Base.showerror(io::IO, err::MethodSubstitutionWarning) =
    print(io, "You called $(err.original), maybe you intended to call $(err.substitute) instead?")

function irgen(ctx::CompilerContext)
    # get the method instance
    isa(ctx.f, Core.Builtin) && throw(CompilerError(ctx, "function is not a generic function"))
    world = typemax(UInt)
    meth = which(ctx.f, ctx.tt)
    sig = Base.signature_type(ctx.f, ctx.tt)::Type
    (ti, env) = ccall(:jl_type_intersection_with_env, Any,
                      (Any, Any), sig, meth.sig)::Core.SimpleVector
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
    method_stack = Vector{Core.MethodInstance}()
    function hook_emit_function(method_instance, code, world)
        push!(method_stack, method_instance)

        # check for recursion
        if method_instance in method_stack[1:end-1]
            throw(CompilerError(ctx, "recursion is currently not supported", backtrace(ctx, method_stack)))
        end

        # check for Base methods that exist in CUDAnative too
        # FIXME: this might be too coarse
        method = method_instance.def
        if Base.moduleroot(method.module) == Base &&
           isdefined(CUDAnative, method_instance.def.name)
            substitute_function = getfield(CUDAnative, method.name)
            tt = Tuple{method_instance.specTypes.parameters[2:end]...}
            if hasmethod(substitute_function, tt)
                method′ = which(substitute_function, tt)
                if Base.moduleroot(method′.module) == CUDAnative
                    @warn "calls to Base intrinsics might be GPU incompatible" exception=(MethodSubstitutionWarning(method, method′), backtrace(ctx, method_stack))
                end
            end
        end
    end
    function hook_emitted_function(method, code, world)
        @assert last(method_stack) == method
        pop!(method_stack)
    end
    params = Base.CodegenParams(cached             = false,
                                track_allocations  = false,
                                code_coverage      = false,
                                static_alloc       = false,
                                prefer_specsig     = true,
                                module_setup       = hook_module_setup,
                                module_activation  = hook_module_activation,
                                raise_exception    = hook_raise_exception,
                                emit_function      = hook_emit_function,
                                emitted_function   = hook_emitted_function)

    # get the code
    mod = let
        ref = ccall(:jl_get_llvmf_defn, LLVM.API.LLVMValueRef,
                    (Any, UInt, Bool, Bool, Base.CodegenParams),
                    linfo, world, #=wrapper=#false, #=optimize=#false, params)
        if ref == C_NULL
            throw(CompilerError(ctx, "the Julia compiler could not generate LLVM IR"))
        end

        llvmf = LLVM.Function(ref)
        LLVM.parent(llvmf)
    end

    # the main module should contain a single jfptr_ function definition,
    # e.g. jfptr_kernel_vadd_62977
    definitions = LLVM.Function[]
    for llvmf in functions(mod)
        if !isdeclaration(llvmf)
            push!(definitions, llvmf)
        end
    end
    wrapper = nothing
    for llvmf in definitions
        if startswith(LLVM.name(llvmf), "jfptr_")
            @assert wrapper == nothing
            wrapper = llvmf
        end
    end
    @assert wrapper != nothing

    # the jfptr wrapper function should point us to the actual entry-point,
    # e.g. julia_kernel_vadd_62984
    entry_tag = let
        m = match(r"jfptr_(.+)_\d+", LLVM.name(wrapper))::RegexMatch
        m.captures[1]
    end
    unsafe_delete!(mod, wrapper)
    entry = let
        re = Regex("julia_$(entry_tag)_\\d+")
        llvmcall_re = Regex("julia_$(entry_tag)_\\d+u\\d+")
        entrypoints = LLVM.Function[]
        for llvmf in definitions
            if llvmf != wrapper
                llvmfn = LLVM.name(llvmf)
                if occursin(re, llvmfn) && !occursin(llvmcall_re, llvmfn)
                    push!(entrypoints, llvmf)
                end
            end
        end
        if length(entrypoints) != 1
            throw(CompilerError(f, tt, cap, "could not find single entry-point";
                                entry=>entry_tag, available=>[LLVM.name.(definitions)]))
        end
        entrypoints[1]
    end

    # link in dependent modules
    for dep in dependencies
        link!(mod, dep)
    end

    # clean up incompatibilities
    for llvmf in functions(mod)
        # only occurs in debug builds
        delete!(function_attributes(llvmf), EnumAttribute("sspstrong", 0, JuliaContext()))

        # dependent modules might have brought in other jfptr wrappers, delete them
        llvmfn = LLVM.name(llvmf)
        if startswith(llvmfn, "jfptr_") && isempty(uses(llvmf))
            unsafe_delete!(mod, llvmf)
            continue
        end

        # llvmcall functions aren't to be called, so mark them internal (cleans up the IR)
        if startswith(llvmfn, "jl_llvmcall")
            linkage!(llvmf, LLVM.API.LLVMInternalLinkage)
            continue
        end

        # make function names safe for ptxas
        # (LLVM ought to do this, see eg. D17738 and D19126), but fails
        # TODO: fix all globals?
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
    alias = something(ctx.alias, String(typeof(something(ctx.inner_f, ctx.f)).name.mt.name))::String
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

    if ctx.kernel
        entry = promote_kernel!(ctx, mod, entry)
    end

    # minimal optimization to get rid of useless generated code (llvmcall, kernel wrapper)
    ModulePassManager() do pm
        add!(pm, ModulePass("ThrowRemoval", remove_throw!))
        always_inliner!(pm)
        run!(pm, mod)
    end

    return mod, entry
end

# HACK: this pass removes `julia_throw_*` functions and replaces them with a `trap`
function remove_throw!(mod::LLVM.Module)
    ctx = LLVM.context(mod)

    # TODO: exit instead of trap?
    trap = if haskey(functions(mod), "llvm.trap")
        functions(mod)["llvm.trap"]
    else
        LLVM.Function(mod, "llvm.trap", LLVM.FunctionType(LLVM.VoidType(ctx)))
    end

    changed = false
    for f in collect(functions(mod))
        fn = LLVM.name(f)
        if startswith(fn, "julia_throw_")   # FIXME: this is coarse
            ft = eltype(llvmtype(f))
            f′ = LLVM.Function(mod, "ptx_"*fn[7:end], ft)

            let builder = Builder(ctx)
                entry = BasicBlock(f′, "entry", ctx)
                position!(builder, entry)
                # TODO: call cuprintf
                call!(builder, trap)
                ret!(builder)
            end
            replace_uses!(f, f′)

            @assert isempty(uses(f))
            unsafe_delete!(mod, f)

            changed = true
        end
    end

    return changed
end

# promote a function to a kernel
# FIXME: sig vs tt (code_llvm vs cufunction)
function promote_kernel!(ctx::CompilerContext, mod::LLVM.Module, entry_f::LLVM.Function)
    kernel = wrap_entry!(ctx, mod, entry_f)

    # property annotations
    # TODO: belongs in irgen? doesn't maxntidx doesn't appear in ptx code?

    annotations = LLVM.Value[kernel]

    ## kernel metadata
    append!(annotations, [MDString("kernel"), ConstantInt(Int32(1), JuliaContext())])

    ## expected CTA sizes
    if ctx.minthreads != nothing
        bounds = CUDAdrv.CuDim3(ctx.minthreads)
        for dim in (:x, :y, :z)
            bound = getfield(bounds, dim)
            append!(annotations, [MDString("reqntid$dim"),
                                  ConstantInt(Int32(bound), JuliaContext())])
        end
    end
    if ctx.maxthreads != nothing
        bounds = CUDAdrv.CuDim3(ctx.maxthreads)
        for dim in (:x, :y, :z)
            bound = getfield(bounds, dim)
            append!(annotations, [MDString("maxntid$dim"),
                                  ConstantInt(Int32(bound), JuliaContext())])
        end
    end

    if ctx.blocks_per_sm != nothing
        append!(annotations, [MDString("minctasm"),
                              ConstantInt(Int32(ctx.blocks_per_sm), JuliaContext())])
    end

    if ctx.maxregs != nothing
        append!(annotations, [MDString("maxnreg"),
                              ConstantInt(Int32(ctx.maxregs), JuliaContext())])
    end


    push!(metadata(mod), "nvvm.annotations", MDNode(annotations))


    return kernel
end

function wrapper_type(julia_t::Type, codegen_t::LLVMType)::LLVMType
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

# generate a kernel wrapper to fix & improve argument passing
function wrap_entry!(ctx::CompilerContext, mod::LLVM.Module, entry_f::LLVM.Function)
    entry_ft = eltype(llvmtype(entry_f)::LLVM.PointerType)::LLVM.FunctionType
    @assert return_type(entry_ft) == LLVM.VoidType(JuliaContext())

    # filter out ghost types, which don't occur in the LLVM function signatures
    sig = Base.signature_type(ctx.f, ctx.tt)::Type
    julia_types = Type[]
    for dt::Type in sig.parameters
        if !isghosttype(dt)
            push!(julia_types, dt)
        end
    end

    # generate the wrapper function type & definition
    wrapper_types = LLVM.LLVMType[wrapper_type(julia_t, codegen_t)
                                  for (julia_t, codegen_t)
                                  in zip(julia_types, parameters(entry_ft))]
    wrapper_fn = replace(LLVM.name(entry_f), r"^.+?_"=>"ptxcall_") # change the CC tag
    wrapper_ft = LLVM.FunctionType(LLVM.VoidType(JuliaContext()), wrapper_types)
    wrapper_f = LLVM.Function(mod, wrapper_fn, wrapper_ft)

    # emit IR performing the "conversions"
    let builder = Builder(JuliaContext())
        entry = BasicBlock(wrapper_f, "entry", JuliaContext())
        position!(builder, entry)

        wrapper_args = Vector{LLVM.Value}()

        # perform argument conversions
        codegen_types = parameters(entry_ft)
        wrapper_params = parameters(wrapper_f)
        param_index = 0
        for (julia_t, codegen_t, wrapper_t, wrapper_param) in
            zip(julia_types, codegen_types, wrapper_types, wrapper_params)
            param_index += 1
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
            else
                push!(wrapper_args, wrapper_param)
                for attr in collect(parameter_attributes(entry_f, param_index))
                    push!(parameter_attributes(wrapper_f, param_index), attr)
                end
            end
        end

        call!(builder, entry_f, wrapper_args)

        ret!(builder)
        dispose(builder)
    end

    # HACK: get rid of invariant.load and const TBAA metadata on loads from pointer args,
    #       since storing to a stack slot violates the semantics of those attributes.
    # TODO: can we emit a wrapper that doesn't violate Julia's metadata?
    for param in parameters(entry_f)
        if isa(llvmtype(param), LLVM.PointerType)
            # collect all uses of the pointer
            worklist = Vector{LLVM.Instruction}(user.(collect(uses(param))))
            while !isempty(worklist)
                value = popfirst!(worklist)

                # remove the invariant.load attribute
                md = metadata(value)
                if haskey(md, LLVM.MD_invariant_load)
                    delete!(md, LLVM.MD_invariant_load)
                end
                if haskey(md, LLVM.MD_tbaa)
                    delete!(md, LLVM.MD_tbaa)
                end

                # recurse on the output of some instructions
                if isa(value, LLVM.BitCastInst) ||
                   isa(value, LLVM.GetElementPtrInst) ||
                   isa(value, LLVM.AddrSpaceCastInst)
                    append!(worklist, user.(collect(uses(value))))
                end

                # IMPORTANT NOTE: if we ever want to inline functions at the LLVM level,
                # we need to recurse into call instructions here, and strip metadata from
                # called functions (see CUDAnative.jl#238).
            end
        end
    end

    # early-inline the original entry function into the wrapper
    push!(function_attributes(entry_f), EnumAttribute("alwaysinline", 0, JuliaContext()))
    linkage!(entry_f, LLVM.API.LLVMInternalLinkage)

    return wrapper_f
end

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

# Optimize a bitcode module according to a certain device capability.
function optimize!(ctx::CompilerContext, mod::LLVM.Module, entry::LLVM.Function)
    tm = machine(ctx.cap, triple(mod))

    let pm = ModulePassManager()
        add_library_info!(pm, triple(mod))
        add_transform_info!(pm, tm)
        internalize!(pm, [LLVM.name(entry)])
        ccall(:jl_add_optimization_passes, Cvoid,
              (LLVM.API.LLVMPassManagerRef, Cint),
              LLVM.ref(pm), Base.JLOptions().opt_level)

        # NVPTX's target machine info enables runtime unrolling,
        # but Julia's pass sequence only invokes the simple unroller.
        loop_unroll!(pm)
        instruction_combining!(pm)  # clean-up redundancy
        licm!(pm)                   # the inner runtime check might be outer loop invariant

        # the above loop unroll pass might have unrolled regular, non-runtime nested loops.
        # that code still needs to be optimized (arguably, multiple unroll passes should be
        # scheduled by the Julia optimizer). do so here, instead of re-optimizing entirely.
        early_csemem_ssa!(pm) # TODO: gvn instead? see NVPTXTargetMachine.cpp::addEarlyCSEOrGVNPass
        dead_store_elimination!(pm)

        # NOTE: if an optimization is missing, try scheduling an entirely new optimization
        # to see which passes need to be added to the list above
        #     LLVM.clopts("-print-after-all", "-filter-print-funcs=$(LLVM.name(entry))")
        #     ModulePassManager() do pm
        #         add_library_info!(pm, triple(mod))
        #         add_transform_info!(pm, tm)
        #         PassManagerBuilder() do pmb
        #             populate!(pm, pmb)
        #         end
        #         run!(pm, mod)
        #     end

        cfgsimplification!(pm)


        ## IPO

        # we compile a module containing the entire call graph,
        # so perform some interprocedural optimizations.

        dead_arg_elimination!(pm)   # parent doesn't use return value --> ret void

        global_optimizer!(pm)
        global_dce!(pm)
        strip_dead_prototypes!(pm)


        run!(pm, mod)
        dispose(pm)
    end
end

function mcgen(ctx::CompilerContext, mod::LLVM.Module, f::LLVM.Function)
    tm = machine(ctx.cap, triple(mod))

    InitializeNVPTXAsmPrinter()
    return String(emit(tm, mod, LLVM.API.LLVMAssemblyFile))
end

# Compile a function to PTX, returning the assembly and an entry point.
# Not to be used directly, see `cufunction` instead.
# FIXME: this pipeline should be partially reusable from eg. code_llvm
#        also, does the kernel argument belong in the compiler context?
function compile_function(ctx::CompilerContext; strip_ir_metadata::Bool=false)
    ## high-level code generation (Julia AST)

    @debug "(Re)compiling function" ctx

    check_method(ctx)


    ## low-level code generation (LLVM IR)

    mod, entry = irgen(ctx)

    @trace("Module entry point: ", LLVM.name(entry))

    # link libdevice, if it is necessary
    libdevice = load_libdevice(ctx)
    for f in functions(mod)
        if isdeclaration(f) && intrinsic_id(f)==0 && haskey(functions(libdevice), LLVM.name(f))
            libdevice_copy = LLVM.Module(libdevice)
            link_libdevice!(ctx, mod, libdevice_copy)
            break
        end
    end

    # optimize the IR
    optimize!(ctx, mod, entry)

    check_invocation(ctx, entry)

    # check generated IR
    check_ir(ctx, mod)
    verify(mod)

    if strip_ir_metadata
        strip_debuginfo!(mod)
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

    ctx = CompilerContext(f, tt, supported_capability(dev), true; kwargs...)

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

function __init_compiler__()
    # enable generation of FMA instructions to mimic behavior of nvcc
    LLVM.clopts("--nvptx-fma-level=1")
end
