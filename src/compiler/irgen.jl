# Julia/LLVM IR generation and transformation passes

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

# NOTE: we remove `throw_...` functions in the ThrowRemoval pass, but that relies on
#       function names, which is fragile. Remove them here as well, to be safe.
const exception_arguments = Vector{LLVM.Value}()
function raise_exception(insblock::BasicBlock, ex::Value)
    fun = LLVM.parent(insblock)
    mod = LLVM.parent(fun)
    ctx = context(mod)

    trap = if haskey(functions(mod), "llvm.trap")
        functions(mod)["llvm.trap"]
    else
        LLVM.Function(mod, "llvm.trap", LLVM.FunctionType(LLVM.VoidType(ctx)))
    end

    let builder = Builder(ctx)
        position!(builder, insblock)

        cuprintf!(builder, "ERROR: an unknown exception occurred$cuprintf_endline")
        call!(builder, trap)

        dispose(builder)
    end

    # mark arguments that passed to `throw` for removal, as they are often GC-allocated.
    push!(exception_arguments, ex)
end

# generate a pseudo-backtrace from a stack of methods being emitted
function backtrace(ctx::CompilerContext, method_stack::Vector{Core.MethodInstance})
    bt = StackTraces.StackFrame[]
    for method_instance in method_stack
        method = method_instance.def
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
    isa(ctx.f, Core.Builtin) && throw(KernelError(ctx, "function is not a generic function"))
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
            throw(KernelError(ctx, "recursion is currently not supported";
                              bt=backtrace(ctx, method_stack)))
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
        @compiler_assert last(method_stack) == method ctx
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
            throw(InternalCompilerError(ctx, "the Julia compiler could not generate LLVM IR"))
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
            @compiler_assert wrapper == nothing ctx first=wrapper second=llvmf
            wrapper = llvmf
        end
    end
    @compiler_assert wrapper != nothing ctx

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
            throw(InternalCompilerError(ctx, "could not find single entry-point";
                                        entry=>entry_tag,
                                        available=>[LLVM.name.(definitions)]))
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

    # HACK: remove unused arguments to exception functions
    for val in exception_arguments
        if isa(val, LLVM.Instruction) && isempty(uses(val))
            bb = LLVM.parent(val)
            unsafe_delete!(bb, val)
        end
    end
    empty!(exception_arguments)

    # rename the entry point
    llvmfn = replace(LLVM.name(entry), r"_\d+$"=>"")
    ## add a friendlier alias
    alias = something(ctx.alias, String(typeof(ctx.f).name.mt.name))::String
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

    # minimal optimization to get rid of useless generated code (llvmcall, kernel wrapper)
    ModulePassManager() do pm
        add!(pm, ModulePass("ThrowRemoval", remove_throw!))
        cfgsimplification!(pm) # makes it easier to find fallthrough control flow
        add!(pm, FunctionPass("ControlFlowFixup", fixup_controlflow!))
        always_inliner!(pm)
        verifier!(pm)
        run!(pm, mod)
    end

    return mod, entry
end

# HACK: this pass replaces `julia_throw_*` void functions with a simple print & trap
#
# this is necessary even though we already have a `raise_exception` hook that just prints,
# as many `throw_...` functions are now `@noinline` which means their arguments survive and
# may cause GC allocations.
#
# TODO: replace with a Cassette-style overdub of the `throw` builtin
function remove_throw!(mod::LLVM.Module)
    ctx = LLVM.context(mod)

    trap = if haskey(functions(mod), "llvm.trap")
        functions(mod)["llvm.trap"]
    else
        LLVM.Function(mod, "llvm.trap", LLVM.FunctionType(LLVM.VoidType(ctx)))
    end

    changed = false

    # NOTE: module pass, since we delete functions
    for f in collect(functions(mod))
        fn = LLVM.name(f)
        ft = eltype(llvmtype(f))

        for re in [# common exceptions as defined in the runtime
                   r"jl_(bounds_error)_.+",
                   # user-code throw functions
                   # FIXME: this is coarse
                   r"julia_throw_(.+)_\d+"]
            m = match(re, fn)
            if m != nothing && return_type(ft) == LLVM.VoidType(ctx)
                ex = m.captures[1]

                # create a function that prints the exception name, and exits
                fn′ = "ptx_throw_$ex"
                f′ = if haskey(functions(mod), fn′)
                    functions(mod)[fn′]
                else
                    ft′ = LLVM.FunctionType(LLVM.VoidType(ctx))
                    f′ = LLVM.Function(mod, fn′, ft′)
                    let builder = Builder(ctx)
                        entry = BasicBlock(f′, "entry", ctx)
                        position!(builder, entry)

                        desc = replace(ex, '_'=>' ')
                        cuprintf!(builder, "ERROR: a $ex exception occurred$cuprintf_endline")
                        call!(builder, trap)
                        ret!(builder)

                        dispose(builder)
                    end
                    f′
                end

                # replace uses of the original function
                for use in uses(f)
                    call = user(use)
                    @assert isa(call, LLVM.CallInst)
                    let builder = Builder(ctx)
                        position!(builder, call)
                        call!(builder, f′)
                        dispose(builder)
                    end
                    unsafe_delete!(LLVM.parent(call), call)
                end

                # remove the original function or declaration
                @assert isempty(uses(f))
                unsafe_delete!(mod, f)


                changed = true
            end
        end
    end

    return changed
end

# HACK: this pass removes control flow that confuses `ptxas`
#
# basically, we only want a single exit from a function. exiting (ret, trap, exit) from
# divergent branches causes ptxas to emit invalid code.
#
# TODO: do this with structured CFG with LLVM?
function fixup_controlflow!(f::LLVM.Function)
    ctx = LLVM.context(f)

    exit_ft = LLVM.FunctionType(LLVM.VoidType(ctx))
    exit = InlineAsm(exit_ft, "exit;", "", true)

    changed = false

    # remove `noreturn` attributes
    #
    # when calling a `noreturn` function, LLVM places an `unreachable` after the call.
    # this leads to an early `ret` from the function.
    attrs = function_attributes(f)
    delete!(attrs, EnumAttribute("noreturn", 0, ctx))

    for bb in blocks(f)
        # remove calls to `trap`, for obvious reasons.
        for inst in instructions(bb)
            if isa(inst, LLVM.CallInst) && LLVM.name(called_value(inst)) == "llvm.trap"
                # replace with call to `exit`
                # FIXME: this still confuses `ptxas`, `brkpt` seems to work but that's fragile
                #let builder = Builder(ctx)
                #    position!(builder, inst)
                #    call!(builder, exit)
                #    dispose(builder)
                #end

                unsafe_delete!(bb, inst)
                changed = true
            end
        end

        # remove `unreachable `terminators.
        #
        # at this point, the optimizer hasn't run, so these hints are placed there by Julia.
        # this happens e.g. with exceptions, and cannot be controlled by a compiler hook.
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

                let builder = Builder(ctx)
                    position!(builder, bb)

                    # find the predecessors to this block
                    predecessors = LLVM.BasicBlock[]
                    for bb′ in blocks(f), inst in instructions(bb′)
                        if isa(inst, LLVM.BrInst)
                            if bb in successors(inst)
                                push!(predecessors, bb′)
                            end
                        end
                    end

                    # find the fall through successors
                    if length(predecessors) == 1
                        predecessor = first(predecessors)
                        br = terminator(predecessor)

                        # find the other successors
                        # NOTE: we've run CFG simplification, so we don't need to handle
                        #       strange IR constructs here.
                        targets = successors(br)
                        if length(targets) == 2
                            for target in targets
                                if target != bb
                                    br!(builder, target)
                                    break
                                end
                            end
                        else
                            @warn "unreachable control flow with $(length(targets))-way branching predecessor"
                            unreachable!(builder)
                        end
                    elseif length(predecessors) > 1
                        @warn "unreachable control flow with multiple predecessors"
                        unreachable!(builder)
                    else
                        # this block has no predecessors, so we can't fall through
                        #
                        # a block without predecessors is either never called,
                        # or the only block in a function.
                        ft = eltype(llvmtype(f))
                        if return_type(ft) == LLVM.VoidType(ctx)
                            # even though returning can lead to invalid control flow,
                            # it mostly happens with functions that just throw,
                            # and leaving the unreachable there would make the optimizer
                            # place another after the call.
                            ret!(builder)
                        else
                            unreachable!(builder)
                        end
                    end

                    dispose(builder)
                end
            end
        end
    end

    return changed
end

