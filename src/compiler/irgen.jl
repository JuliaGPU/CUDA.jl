# LLVM IR generation

function module_setup(mod::LLVM.Module)
    triple!(mod, Int === Int64 ? "nvptx64-nvidia-cuda" : "nvptx-nvidia-cuda")

    # add debug info metadata
    if LLVM.version() >= v"8.0"
        # Set Dwarf Version to 2, the DI printer will downgrade to v2 automatically,
        # but this is technically correct and the only version supported by NVPTX
        LLVM.flags(mod)["Dwarf Version", LLVM.API.LLVMModuleFlagBehaviorWarning] =
            Metadata(ConstantInt(Int32(2), JuliaContext()))
        LLVM.flags(mod)["Debug Info Version", LLVM.API.LLVMModuleFlagBehaviorError] =
            Metadata(ConstantInt(DEBUG_METADATA_VERSION(), JuliaContext()))
    else
        push!(metadata(mod), "llvm.module.flags",
             MDNode([ConstantInt(Int32(1), JuliaContext()),    # llvm::Module::Error
                     MDString("Debug Info Version"),
                     ConstantInt(DEBUG_METADATA_VERSION(), JuliaContext())]))
    end
end

# generate a pseudo-backtrace from a stack of methods being emitted
function backtrace(job::CompilerJob, call_stack::Vector{Core.MethodInstance})
    bt = StackTraces.StackFrame[]
    for method_instance in call_stack
        method = method_instance.def
        if method.name === :overdub && isdefined(method, :generator)
            # The inline frames are maintained by the dwarf based backtrace, but here we only have the
            # calls to overdub directly, the backtrace therefore is collapsed and we have to
            # lookup the overdubbed function, but only if we likely are using the generated variant.
            actual_sig = Tuple{method_instance.specTypes.parameters[3:end]...}
            m = ccall(:jl_gf_invoke_lookup, Any, (Any, UInt), actual_sig, typemax(UInt))
            method = m.func::Method
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
const method_substitution_whitelist = [:hypot]

function compile_method_instance(job::CompilerJob, method_instance::Core.MethodInstance, world)
    function postprocess(ir)
        # get rid of jfptr wrappers
        for llvmf in functions(ir)
            startswith(LLVM.name(llvmf), "jfptr_") && unsafe_delete!(ir, llvmf)
        end

        return
    end

    # set-up the compiler interface
    last_method_instance = nothing
    call_stack = Vector{Core.MethodInstance}()
    dependencies = MultiDict{Core.MethodInstance,LLVM.Function}()
    function hook_module_setup(ref::Ptr{Cvoid})
        ref = convert(LLVM.API.LLVMModuleRef, ref)
        ir = LLVM.Module(ref)
        module_setup(ir)
    end
    function hook_module_activation(ref::Ptr{Cvoid})
        ref = convert(LLVM.API.LLVMModuleRef, ref)
        ir = LLVM.Module(ref)
        postprocess(ir)

        # find the function that this module defines
        llvmfs = filter(llvmf -> !isdeclaration(llvmf) &&
                                 linkage(llvmf) == LLVM.API.LLVMExternalLinkage,
                        collect(functions(ir)))

        llvmf = nothing
        if length(llvmfs) == 1
            llvmf = first(llvmfs)
        elseif length(llvmfs) > 1
            llvmfs = filter!(llvmf -> startswith(LLVM.name(llvmf), "julia_"), llvmfs)
            if length(llvmfs) == 1
                llvmf = first(llvmfs)
            end
        end

        @compiler_assert llvmf !== nothing job

        insert!(dependencies, last_method_instance, llvmf)
    end
    function hook_emit_function(method_instance, code, world)
        push!(call_stack, method_instance)

        # check for recursion
        if method_instance in call_stack[1:end-1]
            throw(KernelError(job, "recursion is currently not supported";
                              bt=backtrace(job, call_stack)))
        end

        # check for Base functions that exist in CUDAnative too
        # FIXME: this might be too coarse
        method = method_instance.def
        if Base.moduleroot(method.module) == Base &&
           isdefined(CUDAnative, method_instance.def.name) &&
           !in(method_instance.def.name, method_substitution_whitelist)
            substitute_function = getfield(CUDAnative, method.name)
            tt = Tuple{method_instance.specTypes.parameters[2:end]...}
            if hasmethod(substitute_function, tt)
                method′ = which(substitute_function, tt)
                if Base.moduleroot(method′.module) == CUDAnative
                    @warn "calls to Base intrinsics might be GPU incompatible" exception=(MethodSubstitutionWarning(method, method′), backtrace(job, call_stack))
                end
            end
        end
    end
    function hook_emitted_function(method, code, world)
        @compiler_assert last(call_stack) == method job
        last_method_instance = pop!(call_stack)
    end
    param_kwargs = [:cached             => false,
                    :track_allocations  => false,
                    :code_coverage      => false,
                    :static_alloc       => false,
                    :prefer_specsig     => true,
                    :module_setup       => hook_module_setup,
                    :module_activation  => hook_module_activation,
                    :emit_function      => hook_emit_function,
                    :emitted_function   => hook_emitted_function]
    if LLVM.version() >= v"8.0" && VERSION >= v"1.3.0-DEV.547"
        push!(param_kwargs, :gnu_pubnames => false)

        debug_info_kind = if Base.JLOptions().debug_level == 0
            LLVM.API.LLVMDebugEmissionKindNoDebug
        elseif Base.JLOptions().debug_level == 1
            LLVM.API.LLVMDebugEmissionKindLineTablesOnly
        elseif Base.JLOptions().debug_level >= 2
            LLVM.API.LLVMDebugEmissionKindFullDebug
        end

        #if CUDAdrv.release() < v"10.2"
            # FIXME: LLVM's debug info crashes CUDA
            # FIXME: this ought to be fixed on 10.2?
            @debug "Incompatibility detected between CUDA and LLVM 8.0+; disabling debug info emission" maxlog=1
            debug_info_kind = LLVM.API.LLVMDebugEmissionKindNoDebug
        #end

        push!(param_kwargs, :debug_info_kind => Cint(debug_info_kind))
    end
    params = Base.CodegenParams(;param_kwargs...)

    # get the code
    ref = ccall(:jl_get_llvmf_defn, LLVM.API.LLVMValueRef,
                (Any, UInt, Bool, Bool, Base.CodegenParams),
                method_instance, world, #=wrapper=#false, #=optimize=#false, params)
    if ref == C_NULL
        throw(InternalCompilerError(job, "the Julia compiler could not generate LLVM IR"))
    end
    llvmf = LLVM.Function(ref)
    ir = LLVM.parent(llvmf)
    postprocess(ir)

    return llvmf, dependencies
end

function irgen(job::CompilerJob, method_instance::Core.MethodInstance, world)
    entry, dependencies = @timeit_debug to "emission" compile_method_instance(job, method_instance, world)
    mod = LLVM.parent(entry)

    # link in dependent modules
    @timeit_debug to "linking" begin
        # we disable Julia's compilation cache not to poison it with GPU-specific code.
        # as a result, we might get multiple modules for a single method instance.
        cache = Dict{String,String}()

        for called_method_instance in keys(dependencies)
            llvmfs = dependencies[called_method_instance]

            # link the first module
            llvmf = popfirst!(llvmfs)
            llvmfn = LLVM.name(llvmf)
            link!(mod, LLVM.parent(llvmf))

            # process subsequent duplicate modules
            for dup_llvmf in llvmfs
                if Base.JLOptions().debug_level >= 2
                    # link them too, to ensure accurate backtrace reconstruction
                    link!(mod, LLVM.parent(dup_llvmf))
                else
                    # don't link them, but note the called function name in a cache
                    dup_llvmfn = LLVM.name(dup_llvmf)
                    cache[dup_llvmfn] = llvmfn
                end
            end
        end

        # resolve function declarations with cached entries
        for llvmf in filter(isdeclaration, collect(functions(mod)))
            llvmfn = LLVM.name(llvmf)
            if haskey(cache, llvmfn)
                def_llvmfn = cache[llvmfn]
                replace_uses!(llvmf, functions(mod)[def_llvmfn])

                @compiler_assert isempty(uses(llvmf)) job
                unsafe_delete!(LLVM.parent(llvmf), llvmf)
            end
        end
    end

    # clean up incompatibilities
    @timeit_debug to "clean-up" for llvmf in functions(mod)
        llvmfn = LLVM.name(llvmf)

        # only occurs in debug builds
        delete!(function_attributes(llvmf), EnumAttribute("sspstrong", 0, JuliaContext()))

        # rename functions
        if !isdeclaration(llvmf)
            # Julia disambiguates local functions by prefixing with `#\d#`.
            # since we don't use a global function namespace, get rid of those tags.
            if occursin(r"^julia_#\d+#", llvmfn)
                llvmfn′ = replace(llvmfn, r"#\d+#"=>"")
                if !haskey(functions(mod), llvmfn′)
                    LLVM.name!(llvmf, llvmfn′)
                    llvmfn = llvmfn′
                end
            end

            # anonymous functions are just named `#\d`, make that somewhat more readable
            m = match(r"_#(\d+)_", llvmfn)
            if m !== nothing
                llvmfn′ = replace(llvmfn, m.match=>"_anonymous$(m.captures[1])_")
                LLVM.name!(llvmf, llvmfn′)
                llvmfn = llvmfn′
            end

            # finally, make function names safe for ptxas
            # (LLVM should to do this, but fails, see eg. D17738 and D19126)
            llvmfn′ = safe_name(llvmfn)
            if llvmfn != llvmfn′
                LLVM.name!(llvmf, llvmfn′)
                llvmfn = llvmfn′
            end
        end
    end

    # add the global exception indicator flag
    emit_exception_flag!(mod)

    # rename the entry point
    if job.name !== nothing
        llvmfn = safe_name(string("julia_", job.name))
    else
        # strip the globalUnique counter
        llvmfn = replace(LLVM.name(entry), r"_\d+$"=>"")
    end
    LLVM.name!(entry, llvmfn)

    # promote entry-points to kernels
    if job.kernel
        entry = promote_kernel!(job, mod, entry)
    end

    # minimal required optimization
    @timeit_debug to "rewrite" ModulePassManager() do pm
        global current_job
        current_job = job

        linkage!(entry, LLVM.API.LLVMExternalLinkage)
        internalize!(pm, [LLVM.name(entry)])

        add!(pm, ModulePass("LowerThrow", lower_throw!))
        add!(pm, FunctionPass("HideUnreachable", hide_unreachable!))
        add!(pm, ModulePass("HideTrap", hide_trap!))
        run!(pm, mod)
    end

    # name mangling
    LLVM.name!(entry, mangle_call(entry, job.tt))

    return mod, entry
end


## name mangling

# we generate function names that look like C++ functions, because many NVIDIA tools
# support them, e.g., grouping different instantiations of the same kernel together.

function mangle_param(t)
    t == Nothing && return "v"

    if isa(t, DataType) || isa(t, Core.Function)
        tn = safe_name(t)
        str = "$(length(tn))$tn"

        if !isempty(t.parameters)
            str *= "I"
            for t in t.parameters
                str *= mangle_param(t)
            end
            str *= "E"
        end

        str
    elseif isa(t, Integer)
        "Li$(t)E"
    else
        tn = safe_name(t)
        "$(length(tn))$tn"
    end
end

function mangle_call(f, tt)
    fn = safe_name(f)
    str = "_Z$(length(fn))$fn"

    for t in tt.parameters
        str *= mangle_param(t)
    end

    return str
end

# make names safe for ptxas
safe_name(fn::String) = replace(fn, r"[^A-Za-z0-9_]"=>"_")
safe_name(f::Union{Core.Function,DataType}) = safe_name(String(nameof(f)))
safe_name(f::LLVM.Function) = safe_name(LLVM.name(f))
safe_name(x) = safe_name(repr(x))


## exception handling

# this pass lowers `jl_throw` and friends to GPU-compatible exceptions.
# this isn't strictly necessary, but has a couple of advantages:
# - we can kill off unused exception arguments that otherwise would allocate or invoke
# - we can fake debug information (lacking a stack unwinder)
#
# once we have thorough inference (ie. discarding `@nospecialize` and thus supporting
# exception arguments) and proper debug info to unwind the stack, this pass can go.
function lower_throw!(mod::LLVM.Module)
    job = current_job::CompilerJob
    changed = false
    @timeit_debug to "lower throw" begin

    throw_functions = Dict{String,String}(
        "jl_throw"                      => "exception",
        "jl_error"                      => "error",
        "jl_too_few_args"               => "too few arguments exception",
        "jl_too_many_args"              => "too many arguments exception",
        "jl_type_error_rt"              => "type error",
        "jl_undefined_var_error"        => "undefined variable error",
        "jl_bounds_error"               => "bounds error",
        "jl_bounds_error_v"             => "bounds error",
        "jl_bounds_error_int"           => "bounds error",
        "jl_bounds_error_tuple_int"     => "bounds error",
        "jl_bounds_error_unboxed_int"   => "bounds error",
        "jl_bounds_error_ints"          => "bounds error",
        "jl_eof_error"                  => "EOF error"
    )

    for (fn, name) in throw_functions
        if haskey(functions(mod), fn)
            f = functions(mod)[fn]

            for use in uses(f)
                call = user(use)::LLVM.CallInst

                # replace the throw with a PTX-compatible exception
                let builder = Builder(JuliaContext())
                    position!(builder, call)
                    emit_exception!(builder, name, call)
                    dispose(builder)
                end

                # remove the call
                call_args = collect(operands(call))[1:end-1] # last arg is function itself
                unsafe_delete!(LLVM.parent(call), call)

                # HACK: kill the exceptions' unused arguments
                for arg in call_args
                    # peek through casts
                    if isa(arg, LLVM.AddrSpaceCastInst)
                        cast = arg
                        arg = first(operands(cast))
                        isempty(uses(cast)) && unsafe_delete!(LLVM.parent(cast), cast)
                    end

                    if isa(arg, LLVM.Instruction) && isempty(uses(arg))
                        unsafe_delete!(LLVM.parent(arg), arg)
                    end
                end

                changed = true
            end

            @compiler_assert isempty(uses(f)) job
         end
     end

    end
    return changed
end

# report an exception in a GPU-compatible manner
#
# the exact behavior depends on the debug level. in all cases, a `trap` will be emitted, On
# debug level 1, the exception name will be printed, and on debug level 2 the individual
# stack frames (as recovered from the LLVM debug information) will be printed as well.
function emit_exception!(builder, name, inst)
    bb = position(builder)
    fun = LLVM.parent(bb)
    mod = LLVM.parent(fun)

    # report the exception
    if Base.JLOptions().debug_level >= 1
        name = globalstring_ptr!(builder, name, "exception")
        if Base.JLOptions().debug_level == 1
            call!(builder, Runtime.get(:report_exception), [name])
        else
            call!(builder, Runtime.get(:report_exception_name), [name])
        end
    end

    # report each frame
    if Base.JLOptions().debug_level >= 2
        rt = Runtime.get(:report_exception_frame)
        bt = backtrace(inst)
        for (i,frame) in enumerate(bt)
            idx = ConstantInt(rt.llvm_types[1], i)
            func = globalstring_ptr!(builder, String(frame.func), "di_func")
            file = globalstring_ptr!(builder, String(frame.file), "di_file")
            line = ConstantInt(rt.llvm_types[4], frame.line)
            call!(builder, rt, [idx, func, file, line])
        end
    end

    # signal the exception
    call!(builder, Runtime.get(:signal_exception))

    trap = if haskey(functions(mod), "llvm.trap")
        functions(mod)["llvm.trap"]
    else
        LLVM.Function(mod, "llvm.trap", LLVM.FunctionType(LLVM.VoidType(JuliaContext())))
    end
    call!(builder, trap)
end

# HACK: this pass removes `unreachable` information from LLVM
#
# `ptxas` is buggy and cannot deal with thread-divergent control flow in the presence of
# shared memory (see JuliaGPU/CUDAnative.jl#4). avoid that by rewriting control flow to fall
# through any other block. this is semantically invalid, but the code is unreachable anyhow
# (and we expect it to be preceded by eg. a noreturn function, or a trap).
#
# TODO: can LLVM do this with structured CFGs? It seems to have some support, but seemingly
#       only to prevent introducing non-structureness during optimization (ie. the front-end
#       is still responsible for generating structured control flow).
function hide_unreachable!(fun::LLVM.Function)
    job = current_job::CompilerJob
    changed = false
    @timeit_debug to "hide unreachable" begin

    # remove `noreturn` attributes
    #
    # when calling a `noreturn` function, LLVM places an `unreachable` after the call.
    # this leads to an early `ret` from the function.
    attrs = function_attributes(fun)
    delete!(attrs, EnumAttribute("noreturn", 0, JuliaContext()))

    # build a map of basic block predecessors
    predecessors = Dict(bb => Set{LLVM.BasicBlock}() for bb in blocks(fun))
    @timeit_debug to "predecessors" for bb in blocks(fun)
        insts = instructions(bb)
        if !isempty(insts)
            inst = last(insts)
            if isterminator(inst)
                for bb′ in successors(inst)
                    push!(predecessors[bb′], bb)
                end
            end
        end
    end

    # scan for unreachable terminators and alternative successors
    worklist = Pair{LLVM.BasicBlock, Union{Nothing,LLVM.BasicBlock}}[]
    @timeit_debug to "find" for bb in blocks(fun)
        unreachable = terminator(bb)
        if isa(unreachable, LLVM.UnreachableInst)
            unsafe_delete!(bb, unreachable)
            changed = true

            try
                terminator(bb)
                # the basic-block is still terminated properly, nothing to do
                # (this can happen with `ret; unreachable`)
                # TODO: `unreachable; unreachable`
            catch ex
                isa(ex, UndefRefError) || rethrow(ex)
                let builder = Builder(JuliaContext())
                    position!(builder, bb)

                    # find the strict predecessors to this block
                    preds = collect(predecessors[bb])

                    # find a fallthrough block: recursively look at predecessors
                    # and find a successor that branches to any other block
                    fallthrough = nothing
                    while !isempty(preds)
                        # find an alternative successor
                        for pred in preds, succ in successors(terminator(pred))
                            if succ != bb
                                fallthrough = succ
                                break
                            end
                        end
                        fallthrough === nothing || break

                        # recurse upwards
                        old_preds = copy(preds)
                        empty!(preds)
                        for pred in old_preds
                            append!(preds, predecessors[pred])
                        end
                    end
                    push!(worklist, bb => fallthrough)

                    dispose(builder)
                end
            end
        end
    end

    # apply the pending terminator rewrites
    @timeit_debug to "replace" if !isempty(worklist)
        let builder = Builder(JuliaContext())
            for (bb, fallthrough) in worklist
                position!(builder, bb)
                if fallthrough !== nothing
                    br!(builder, fallthrough)
                else
                    # couldn't find any other successor. this happens with functions
                    # that only contain a single block, or when the block is dead.
                    ft = eltype(llvmtype(fun))
                    if return_type(ft) == LLVM.VoidType(JuliaContext())
                        # even though returning can lead to invalid control flow,
                        # it mostly happens with functions that just throw,
                        # and leaving the unreachable there would make the optimizer
                        # place another after the call.
                        ret!(builder)
                    else
                        unreachable!(builder)
                    end
                end
            end
        end
    end

    end
    return changed
end

# HACK: this pass removes calls to `trap` and replaces them with inline assembly
#
# if LLVM knows we're trapping, code is marked `unreachable` (see `hide_unreachable!`).
function hide_trap!(mod::LLVM.Module)
    job = current_job::CompilerJob
    changed = false
    @timeit_debug to "hide trap" begin

    # inline assembly to exit a thread, hiding control flow from LLVM
    exit_ft = LLVM.FunctionType(LLVM.VoidType(JuliaContext()))
    exit = if job.cap < v"7"
        # ptxas for old compute capabilities has a bug where it messes up the
        # synchronization stack in the presence of shared memory and thread-divergend exit.
        InlineAsm(exit_ft, "trap;", "", true)
    else
        InlineAsm(exit_ft, "exit;", "", true)
    end

    if haskey(functions(mod), "llvm.trap")
        trap = functions(mod)["llvm.trap"]

        for use in uses(trap)
            val = user(use)
            if isa(val, LLVM.CallInst)
                let builder = Builder(JuliaContext())
                    position!(builder, val)
                    call!(builder, exit)
                    dispose(builder)
                end
                unsafe_delete!(LLVM.parent(val), val)
                changed = true
            end
        end
    end

    end
    return changed
end

# emit a global variable for storing the current exception status
#
# since we don't actually support globals, access to this variable is done by calling the
# cudanativeExceptionFlag function (lowered here to actual accesses of the variable)
function emit_exception_flag!(mod::LLVM.Module)
    # add the global variable
    T_ptr = convert(LLVMType, Ptr{Cvoid})
    gv = GlobalVariable(mod, T_ptr, "exception_flag")
    initializer!(gv, LLVM.ConstantInt(T_ptr, 0))
    linkage!(gv, LLVM.API.LLVMWeakAnyLinkage)
    extinit!(gv, true)

    # lower uses of the getter
    if haskey(functions(mod), "cudanativeExceptionFlag")
        buf_getter = functions(mod)["cudanativeExceptionFlag"]
        @assert return_type(eltype(llvmtype(buf_getter))) == eltype(llvmtype(gv))

        # find uses
        worklist = Vector{LLVM.CallInst}()
        for use in uses(buf_getter)
            call = user(use)::LLVM.CallInst
            push!(worklist, call)
        end

        # replace uses by a load from the global variable
        for call in worklist
            Builder(JuliaContext()) do builder
                position!(builder, call)
                ptr = load!(builder, gv)
                replace_uses!(call, ptr)
            end
            unsafe_delete!(LLVM.parent(call), call)
        end
    end
end


## kernel promotion

# promote a function to a kernel
# FIXME: sig vs tt (code_llvm vs cufunction)
function promote_kernel!(job::CompilerJob, mod::LLVM.Module, entry_f::LLVM.Function)
    kernel = wrap_entry!(job, mod, entry_f)

    # property annotations
    # TODO: belongs in irgen? maxntidx doesn't appear in ptx code?

    annotations = LLVM.Value[kernel]

    ## kernel metadata
    append!(annotations, [MDString("kernel"), ConstantInt(Int32(1), JuliaContext())])

    ## expected CTA sizes
    if job.minthreads != nothing
        bounds = CUDAdrv.CuDim3(job.minthreads)
        for dim in (:x, :y, :z)
            bound = getfield(bounds, dim)
            append!(annotations, [MDString("reqntid$dim"),
                                  ConstantInt(Int32(bound), JuliaContext())])
        end
    end
    if job.maxthreads != nothing
        bounds = CUDAdrv.CuDim3(job.maxthreads)
        for dim in (:x, :y, :z)
            bound = getfield(bounds, dim)
            append!(annotations, [MDString("maxntid$dim"),
                                  ConstantInt(Int32(bound), JuliaContext())])
        end
    end

    if job.blocks_per_sm != nothing
        append!(annotations, [MDString("minctasm"),
                              ConstantInt(Int32(job.blocks_per_sm), JuliaContext())])
    end

    if job.maxregs != nothing
        append!(annotations, [MDString("maxnreg"),
                              ConstantInt(Int32(job.maxregs), JuliaContext())])
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
function wrap_entry!(job::CompilerJob, mod::LLVM.Module, entry_f::LLVM.Function)
    entry_ft = eltype(llvmtype(entry_f)::LLVM.PointerType)::LLVM.FunctionType
    @compiler_assert return_type(entry_ft) == LLVM.VoidType(JuliaContext()) job

    # filter out ghost types, which don't occur in the LLVM function signatures
    sig = Base.signature_type(job.f, job.tt)::Type
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

    # Get debuginfo from the entry_f and copy it to the wrapper_f
    # on -g0 julia does not add a DICU and DISP object
    if LLVM.version() >= v"8.0"
        if Base.JLOptions().debug_level > 0
            SP = LLVM.get_subprogram(entry_f)
            LLVM.set_subprogram!(wrapper_f, SP)
        end
    end

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
                @compiler_assert isa(codegen_t, LLVM.PointerType) job
                @compiler_assert eltype(codegen_t) == wrapper_t job

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

    # early-inline the original entry function into the wrapper
    push!(function_attributes(entry_f), EnumAttribute("alwaysinline", 0, JuliaContext()))
    linkage!(entry_f, LLVM.API.LLVMInternalLinkage)

    fixup_metadata!(entry_f)
    ModulePassManager() do pm
        always_inliner!(pm)
        run!(pm, mod)
    end

    return wrapper_f
end

# HACK: get rid of invariant.load and const TBAA metadata on loads from pointer args,
#       since storing to a stack slot violates the semantics of those attributes.
# TODO: can we emit a wrapper that doesn't violate Julia's metadata?
function fixup_metadata!(f::LLVM.Function)
    for param in parameters(f)
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
end
