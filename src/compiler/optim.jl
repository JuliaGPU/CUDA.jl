# LLVM IR optimization

function optimize!(job::CompilerJob, mod::LLVM.Module, entry::LLVM.Function)
    tm = machine(job.cap, triple(mod))

    function initialize!(pm)
        add_library_info!(pm, triple(mod))
        add_transform_info!(pm, tm)
    end

    global current_job
    current_job = job

    # Julia-specific optimizations
    #
    # NOTE: we need to use multiple distinct pass managers to force pass ordering;
    #       intrinsics should never get lowered before Julia has optimized them.
    if VERSION < v"1.2.0-DEV.375"
        # with older versions of Julia, intrinsics are lowered unconditionally so we need to
        # replace them with GPU-compatible counterparts before anything else. that breaks
        # certain optimizations though: https://github.com/JuliaGPU/CUDAnative.jl/issues/340

        ModulePassManager() do pm
            initialize!(pm)
            add!(pm, FunctionPass("LowerGCFrame", lower_gc_frame!))
            aggressive_dce!(pm) # remove dead uses of ptls
            add!(pm, ModulePass("LowerPTLS", lower_ptls!))
            run!(pm, mod)
        end

        ModulePassManager() do pm
            initialize!(pm)
            ccall(:jl_add_optimization_passes, Cvoid,
                    (LLVM.API.LLVMPassManagerRef, Cint),
                    LLVM.ref(pm), Base.JLOptions().opt_level)
            run!(pm, mod)
        end
    else
        ModulePassManager() do pm
            initialize!(pm)
            ccall(:jl_add_optimization_passes, Cvoid,
                    (LLVM.API.LLVMPassManagerRef, Cint, Cint),
                    LLVM.ref(pm), Base.JLOptions().opt_level, #=lower_intrinsics=# 0)
            run!(pm, mod)
        end

        ModulePassManager() do pm
            initialize!(pm)

            # lower intrinsics
            add!(pm, FunctionPass("LowerGCFrame", lower_gc_frame!))
            aggressive_dce!(pm) # remove dead uses of ptls
            add!(pm, ModulePass("LowerPTLS", lower_ptls!))

            # the Julia GC lowering pass also has some clean-up that is required
            if VERSION >= v"1.2.0-DEV.531"
                late_lower_gc_frame!(pm)
            end

            run!(pm, mod)
        end
    end

    # PTX-specific optimizations
    ModulePassManager() do pm
        initialize!(pm)

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

        # get rid of the internalized functions; now possible unused
        global_dce!(pm)

        run!(pm, mod)
    end

    # we compile a module containing the entire call graph,
    # so perform some interprocedural optimizations.
    #
    # for some reason, these passes need to be distinct from the regular optimization chain,
    # or certain values (such as the constant arrays used to populare llvm.compiler.user ad
    # part of the LateLowerGCFrame pass) aren't collected properly.
    #
    # these might not always be safe, as Julia's IR metadata isn't designed for IPO.
    ModulePassManager() do pm
        dead_arg_elimination!(pm)   # parent doesn't use return value --> ret void

        run!(pm, mod)
    end

    return entry
end


## lowering intrinsics

# lower object allocations to to PTX malloc
#
# this is a PoC implementation that is very simple: allocate, and never free. it also runs
# _before_ Julia's GC lowering passes, so we don't get to use the results of its analyses.
# when we ever implement a more potent GC, we will need those results, but the relevant pass
# is currently very architecture/CPU specific: hard-coded pool sizes, TLS references, etc.
# such IR is hard to clean-up, so we probably will need to have the GC lowering pass emit
# lower-level intrinsics which then can be lowered to architecture-specific code.
function lower_gc_frame!(fun::LLVM.Function)
    job = current_job::CompilerJob
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

        @compiler_assert isempty(uses(alloc_obj)) job
    end

    # we don't care about write barriers
    if haskey(functions(mod), "julia.write_barrier")
        barrier = functions(mod)["julia.write_barrier"]

        for use in uses(barrier)
            call = user(use)::LLVM.CallInst
            unsafe_delete!(LLVM.parent(call), call)
            changed = true
        end

        @compiler_assert isempty(uses(barrier)) job
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
    job = current_job::CompilerJob
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

        @compiler_assert isempty(uses(ptls_getter)) job
     end

    return changed
end
