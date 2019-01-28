# LLVM IR optimization

function optimize!(ctx::CompilerContext, mod::LLVM.Module, entry::LLVM.Function)
    tm = machine(ctx.cap, triple(mod))

    if ctx.kernel
        entry = promote_kernel!(ctx, mod, entry)
    end

    let pm = ModulePassManager()
        global global_ctx
        global_ctx = ctx

        add_library_info!(pm, triple(mod))
        add_transform_info!(pm, tm)
        internalize!(pm, [LLVM.name(entry)])

        # lower intrinsics
        add!(pm, FunctionPass("LowerGCFrame", lower_gc_frame!))
        aggressive_dce!(pm) # remove dead uses of ptls
        add!(pm, ModulePass("LowerPTLS", lower_ptls!))

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

        constant_merge!(pm)

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

        run!(pm, mod)
        dispose(pm)
    end

    # we compile a module containing the entire call graph,
    # so perform some interprocedural optimizations.
    #
    # for some reason, these passes need to be distinct from the regular optimization chain,
    # or certain values (such as the constant arrays used to populare llvm.compiler.user ad
    # part of the LateLowerGCFrame pass) aren't collected properly.
    #
    # these might not always be safe, as Julia's IR metadata isn't designed for IPO.
    let pm = ModulePassManager()
        dead_arg_elimination!(pm)   # parent doesn't use return value --> ret void

        run!(pm, mod)
        dispose(pm)
    end

    return entry
end


## kernel-specific optimizations

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
    @compiler_assert return_type(entry_ft) == LLVM.VoidType(JuliaContext()) ctx

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
                @compiler_assert isa(codegen_t, LLVM.PointerType) ctx
                @compiler_assert eltype(codegen_t) == wrapper_t ctx

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
        verifier!(pm)
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

# lower object allocations to to PTX malloc
#
# this is a PoC implementation that is very simple: allocate, and never free. it also runs
# _before_ Julia's GC lowering passes, so we don't get to use the results of its analyses.
# when we ever implement a more potent GC, we will need those results, but the relevant pass
# is currently very architecture/CPU specific: hard-coded pool sizes, TLS references, etc.
# such IR is hard to clean-up, so we probably will need to have the GC lowering pass emit
# lower-level intrinsics which then can be lowered to architecture-specific code.
function lower_gc_frame!(fun::LLVM.Function)
    ctx = global_ctx::CompilerContext
    mod = LLVM.parent(fun)
    changed = false

    # plain alloc
    if haskey(functions(mod), "julia.gc_alloc_obj")
        alloc_obj = functions(mod)["julia.gc_alloc_obj"]
        alloc_obj_ft = eltype(llvmtype(alloc_obj))
        T_prjlvalue = return_type(alloc_obj_ft)
        T_pjlvalue = convert(LLVMType, Any, true)

        for use in uses(alloc_obj)
            call = user(use)::LLVM.CallInst

            # decode the call
            ops = collect(operands(call))
            sz = ops[2]

            # replace with PTX alloc_obj
            let builder = Builder(JuliaContext())
                position!(builder, call)
                ptr = call!(builder, Runtime.get(:gc_pool_alloc), [sz])
                replace_uses!(call, ptr)
                dispose(builder)
            end

            unsafe_delete!(LLVM.parent(call), call)

            changed = true
        end

        @compiler_assert isempty(uses(alloc_obj)) ctx
    end

    # we don't care about write barriers
    if haskey(functions(mod), "julia.write_barrier")
        barrier = functions(mod)["julia.write_barrier"]

        for use in uses(barrier)
            call = user(use)::LLVM.CallInst
            unsafe_delete!(LLVM.parent(call), call)
            changed = true
        end

        @compiler_assert isempty(uses(barrier)) ctx
    end

    return changed
end

# lower the `julia.ptls_states` intrinsic by removing it, since it is GPU incompatible.
#
# this assumes and checks that the TLS is unused, which should be the case for most GPU code
# after lowering the GC intrinsics to TLS-less code and having run DCE.
#
# TODO: maybe don't have Julia emit actual uses of the TLS, but use intrinsics instead,
#       making it easier to remove or reimplement that functionality here.
function lower_ptls!(mod::LLVM.Module)
    ctx = global_ctx::CompilerContext
    changed = false

    if haskey(functions(mod), "julia.ptls_states")
        ptls_getter = functions(mod)["julia.ptls_states"]

        for use in uses(ptls_getter)
            val = user(use)
            if !isempty(uses(val))
                error("Thread local storage is not implemented")
            end
            unsafe_delete!(LLVM.parent(val), val)
            changed = true
        end

        @compiler_assert isempty(uses(ptls_getter)) ctx
     end

    return changed
end
