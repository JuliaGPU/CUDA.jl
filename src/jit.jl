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

    # add debug info metadata
    push!(metadata(mod), "llvm.module.flags",
         MDNode([ConstantInt(Int32(1)),    # llvm::Module::Error
                 MDString("Debug Info Version"),
                 ConstantInt(DEBUG_METADATA_VERSION())]))
end

full_fn(f::Core.Function, tt=Tuple{}) = "$(typeof(f).name.mt.name)($(join(tt.parameters, ", ")))"

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

function irgen(@nospecialize(f), @nospecialize(tt))
    # get the method instance
    isa(f, Core.Builtin) && throw(ArgumentError("argument is not a generic function"))
    world = typemax(UInt)
    meth = which(f, tt)
    sig_tt = Tuple{typeof(f), tt.parameters...}
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
            error("could not compile the specified method")
        end

        llvmf = LLVM.Function(ref)
        LLVM.parent(llvmf)
    end

    # the main module should contain a single jfptr_ function definition,
    # e.g. jlcall_kernel_vadd_62977
    definitions = filter(f->!isdeclaration(f), functions(mod))
    wrapper = let
        fs = if VERSION >= v"0.7.0-DEV.4747"
            collect(filter(f->startswith(LLVM.name(f), "jfptr_"), definitions))
        else
            collect(filter(f->startswith(LLVM.name(f), "jlcall_"), definitions))
        end
        @assert length(fs) == 1
        fs[1]
    end

    # the jlcall wrapper function should point us to the actual entry-point,
    # e.g. julia_kernel_vadd_62984
    entry_tag = let
        m = if VERSION >= v"0.7.0-DEV.4747"
            match(r"jfptr_(.+)_\d+", LLVM.name(wrapper))
        else
            match(r"jlcall_(.+)_\d+", LLVM.name(wrapper))
        end
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
            error("Could not find single entry-point for $entry_tag (available functions: ",
                  join(map(f->LLVM.name(f), definitions), ", "), ")")
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

    return mod, entry
end

# promote a function to a kernel
function promote_kernel!(mod::LLVM.Module, entry_f::LLVM.Function, @nospecialize(tt);
                         minthreads::Union{Nothing,CuDim}=nothing,
                         maxthreads::Union{Nothing,CuDim}=nothing,
                         blocks_per_sm::Union{Nothing,Integer}=nothing,
                         maxregs::Union{Nothing,Integer}=nothing,
                         name::Union{Nothing,String}=nothing)
    if name === nothing
        name = replace(LLVM.name(entry_f)[7:end], r"_\d+$" => "")
    end
    kernel = wrap_entry!(mod, entry_f, tt, name)


    # property annotations
    # TODO: belongs in irgen? doesn't maxntidx doesn't appear in ptx code?

    annotations = LLVM.Value[kernel]

    ## kernel metadata
    append!(annotations, [MDString("kernel"), ConstantInt(Int32(1))])

    ## expected CTA sizes
    for (typ,vals) in (:req=>minthreads, :max=>maxthreads)
        if vals != nothing
            bounds = CUDAdrv.CuDim3(vals)
            for dim in (:x, :y, :z)
                bound = getfield(bounds, dim)
                append!(annotations, [MDString("$(typ)ntid$(dim)"),
                                      ConstantInt(Int32(bound))])
            end
        end
    end

    if blocks_per_sm != nothing
        append!(annotations, [MDString("minctasm"), ConstantInt(Int32(blocks_per_sm))])
    end

    if maxregs != nothing
        append!(annotations, [MDString("maxnreg"), ConstantInt(Int32(maxregs))])
    end


    push!(metadata(mod), "nvvm.annotations", MDNode(annotations))


    return kernel
end

# maintain our own "global unique" suffix for disambiguating kernels
globalUnique = 0

# generate a kernel wrapper to fix & improve argument passing
function wrap_entry!(mod::LLVM.Module, entry_f::LLVM.Function, @nospecialize(tt), name)
    entry_ft = eltype(llvmtype(entry_f))
    @assert return_type(entry_ft) == LLVM.VoidType(jlctx[])

    # filter out ghost types, which don't occur in the LLVM function signatures
    julia_types = filter(dt->!isghosttype(dt), tt.parameters)

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
    wrapper_fn = "ptxcall_$(name)_$(globalUnique+=1)"
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
function link_libdevice!(mod::LLVM.Module, cap::VersionNumber)
    CUDAnative.configured || return

    # find libdevice
    path = if isa(libdevice, Dict)
        # select the most recent & compatible library
        vers = keys(CUDAnative.libdevice)
        compat_vers = Set(ver for ver in vers if ver <= cap)
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
function optimize!(mod::LLVM.Module, entry::LLVM.Function, cap::VersionNumber)
    tm = machine(cap, triple(mod))

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

function mcgen(mod::LLVM.Module, f::LLVM.Function, cap::VersionNumber)
    tm = machine(cap, triple(mod))

    InitializeNVPTXAsmPrinter()
    return String(emit(tm, mod, LLVM.API.LLVMAssemblyFile))
end

# Compile a function to PTX, returning the assembly and an entry point.
# Not to be used directly, see `cufunction` instead.
#
# The `kernel` argument indicates whether we are compiling a kernel entry-point function,
# in which case extra metadata needs to be attached.
function compile_function(@nospecialize(f), @nospecialize(tt), cap::VersionNumber;
                          kernel::Bool=true, kwargs...)
    ## high-level code generation (Julia AST)

    fn = full_fn(f, tt)
    @debug "(Re)compiling function" f tt cap

    check_invocation(f, tt; kernel=kernel)
    sig = Base.signature_type(f, tt)


    ## low-level code generation (LLVM IR)

    mod, entry = irgen(f, tt)
    if kernel
        entry = promote_kernel!(mod, entry, sig; kwargs...)
    end
    @trace("Module entry point: ", LLVM.name(entry))

    # link libdevice, if it might be necessary
    # TODO: should be more find-grained -- only matching functions actually in this libdevice
    if any(f->isdeclaration(f) && intrinsic_id(f)==0, functions(mod))
        link_libdevice!(mod, cap)
    end

    # optimize the IR (otherwise the IR isn't necessarily compatible)
    optimize!(mod, entry, cap)

    # make sure any non-isbits arguments are unused
    real_arg_i = 0
    for (arg_i,dt) in enumerate(sig.parameters)
        isghosttype(dt) && continue
        real_arg_i += 1

        if !isbitstype(dt)
            param = parameters(entry)[real_arg_i]
            if !isempty(uses(param))
                throw(ArgumentError("Passing and using non-bitstype argument $arg_i of type $dt"))
            end
        end
    end

    # validate generated IR
    errors = validate_ir(mod)
    if !isempty(errors)
        for e in errors
            @warn("Encountered incompatible LLVM IR for $fn", e)
        end
        error("LLVM IR generated for $fn at capability $cap is not compatible")
    end


    ## machine code generation (PTX assembly)

    module_asm = mcgen(mod, entry, cap)

    return module_asm, LLVM.name(entry)
end

# check validity of a function invocation, specified by the generic function and a tupletype
function check_invocation(@nospecialize(f), @nospecialize(tt); kernel::Bool=false)
    fn = full_fn(f, tt)

    # get the method
    ms = Base.methods(f, tt)
    isempty(ms)   && throw(ArgumentError("no method found for $fn"))
    length(ms)!=1 && throw(ArgumentError("no unique matching method for $fn"))
    m = first(ms)

    # kernels can't return values
    if kernel
        rt = Base.return_types(f, tt)[1]
        if rt != Nothing
            throw(ArgumentError("$fn is not a valid kernel as it returns $rt"))
        end
    end
end

# (f::Function, tt::Type, cap::VersionNumber; kwargs...)
const compile_hook = Ref{Union{Nothing,Function}}(nothing)

# Main entry point for compiling a Julia function + argtypes to a callable CUDA function
function cufunction(dev::CuDevice, @nospecialize(f), @nospecialize(inner_f), @nospecialize(tt);
                    name=nothing, kwargs...)
    CUDAnative.configured || error("CUDAnative.jl has not been configured; cannot JIT code.")
    @assert isa(f, Core.Function)

    # select a capability level
    dev_cap = capability(dev)
    compat_caps = filter(cap -> cap <= dev_cap, target_support)
    isempty(compat_caps) &&
        error("Device capability v$dev_cap not supported by available toolchain")
    cap = maximum(compat_caps)

    if compile_hook[] != nothing
        global globalUnique
        previous_globalUnique = globalUnique
        compile_hook[](f, inner_f, tt, cap; kwargs...)
        globalUnique = previous_globalUnique
    end

    if name === nothing
        # if the user didn't specify a compiler kernel name,
        # try to figure one out from the inner function
        fn = String(typeof(inner_f).name.mt.name)
        if occursin('#', fn)
            name = "anonymous"
        else
            name = safe_fn(fn)
        end
    end

    (module_asm, module_entry) = compile_function(f, tt, cap; name=name, kwargs...)

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
