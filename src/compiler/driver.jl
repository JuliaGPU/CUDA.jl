# compiler driver and main interface

# (::CompilerJob)
const compile_hook = Ref{Union{Nothing,Function}}(nothing)

"""
    compile(target::Symbol, cap::VersionNumber, f, tt, kernel=true;
            optimize=true, strip=false, ...)

Compile a function `f` invoked with types `tt` for device capability `cap` to one of the
following formats as specified by the `target` argument: `:julia` for Julia IR, `:llvm` for
LLVM IR, `:ptx` for PTX assembly and `:cuda` for CUDA driver objects. If the `kernel` flag
is set, specialized code generation and optimization for kernel functions is enabled.

The following keyword arguments are supported:
- `optimize`: optimize the code (default: true)
- `strip`: strip non-functional metadata and debug information  (default: false)

Other keyword arguments can be found in the documentation of [`cufunction`](@ref).
"""
compile(target::Symbol, cap::VersionNumber, @nospecialize(f::Core.Function), @nospecialize(tt),
        kernel::Bool=true; optimize::Bool=true, strip::Bool=false, kwargs...) =
    compile(target, CompilerJob(f, tt, cap, kernel; kwargs...); optimize=optimize, strip=strip)

function compile(target::Symbol, job::CompilerJob;
                 libraries::Bool=true, optimize::Bool=true, strip::Bool=false)
    @debug "(Re)compiling function" job

    if compile_hook[] != nothing
        global globalUnique
        previous_globalUnique = globalUnique

        compile_hook[](job)

        globalUnique = previous_globalUnique
    end

    return codegen(target, job; libraries=libraries, optimize=optimize, strip=strip)
end

function codegen(target::Symbol, job::CompilerJob; libraries::Bool=true,
                 optimize::Bool=true, strip::Bool=false)
    ## Julia IR

    @timeit to[] "Julia front-end" begin
        check_method(job)

        # get the method instance
        world = typemax(UInt)
        meth = which(job.f, job.tt)
        sig = Base.signature_type(job.f, job.tt)::Type
        (ti, env) = ccall(:jl_type_intersection_with_env, Any,
                          (Any, Any), sig, meth.sig)::Core.SimpleVector
        if VERSION >= v"1.2.0-DEV.320"
            meth = Base.func_for_method_checked(meth, ti, env)
        else
            meth = Base.func_for_method_checked(meth, ti)
        end
        method_instance = ccall(:jl_specializations_get_linfo, Ref{Core.MethodInstance},
                      (Any, Any, Any, UInt), meth, ti, env, world)

        for var in env
            if var isa TypeVar
                throw(KernelError(job, "method captures a typevar (you probably use an unbound type variable)"))
            end
        end
    end

    target == :julia && return method_instance


    ## LLVM IR

    # always preload the runtime, and do so early; it cannot be part of any timing block
    # because it recurses into the compiler
    if libraries
        runtime = load_runtime(job.cap)
        runtime_defs = LLVM.name.(filter(f -> !isdeclaration(f),
                                         collect(functions(runtime))))
    end

    @timeit to[] "LLVM middle-end" begin
        ir, kernel = @timeit to[] "IR generation" irgen(job, method_instance, world)

        if libraries
            # find out if there's any libraries we actually need
            decls = LLVM.name.(filter(f -> isdeclaration(f) &&
                                           intrinsic_id(f) == 0,
                                      collect(functions(ir))))
            need_runtime = any(fn -> fn in runtime_defs, decls)
            need_libdevice = any(fn->startswith(fn, "__nv_"), decls)

            if need_libdevice
                libdevice = load_libdevice(job.cap)
                @timeit to[] "device library" if need_library(libdevice)
                    link_libdevice!(job, ir, libdevice)
                end
            end
        end

        if optimize
            kernel = @timeit to[] "optimization" optimize!(job, ir, kernel)
        end

        if libraries
            @timeit to[] "runtime library" if need_runtime
                link_library!(job, ir, runtime)
            end
        end

        if ccall(:jl_is_debugbuild, Cint, ()) == 1
            @timeit to[] "verification" verify(ir)
        end

        if strip
            @timeit to[] "strip debug info" strip_debuginfo!(ir)
        end

        kernel_fn = LLVM.name(kernel)
    end

    target == :llvm && return ir, kernel


    ## dynamic parallelism

    kernels = OrderedDict{CompilerJob, String}(job => kernel_fn)

    if haskey(functions(ir), "cudanativeCompileKernel")
        dyn_maker = functions(ir)["cudanativeCompileKernel"]

        # iterative compilation (non-recursive)
        changed = true
        while changed
            changed = false

            # find dynamic kernel invocations
            # TODO: recover this information earlier, from the Julia IR
            worklist = MultiDict{CompilerJob, LLVM.CallInst}()
            for use in uses(dyn_maker)
                # decode the call
                call = user(use)::LLVM.CallInst
                id = convert(Int, first(operands(call)))

                global delayed_cufunctions
                dyn_f, dyn_tt = delayed_cufunctions[id]
                dyn_job = CompilerJob(dyn_f, dyn_tt, job.cap, #=kernel=# true)
                push!(worklist, dyn_job => call)
            end

            # compile and link
            for dyn_job in keys(worklist)
                # cached compilation
                dyn_kernel_fn = get!(kernels, dyn_job) do
                    dyn_ir, dyn_kernel = codegen(:llvm, dyn_job; optimize=optimize, strip=strip)
                    dyn_kernel_fn = LLVM.name(dyn_kernel)
                    link!(ir, dyn_ir)
                    changed = true
                    dyn_kernel_fn
                end
                dyn_kernel = functions(ir)[dyn_kernel_fn]

                # insert a pointer to the function everywhere the kernel is used
                T_ptr = convert(LLVMType, Ptr{Cvoid})
                for call in worklist[dyn_job]
                    Builder(JuliaContext()) do builder
                        position!(builder, call)
                        fptr = ptrtoint!(builder, dyn_kernel, T_ptr)
                        replace_uses!(call, fptr)
                    end
                    unsafe_delete!(LLVM.parent(call), call)
                end
            end
        end

        # all dynamic launches should have been resolved
        @compiler_assert isempty(uses(dyn_maker)) job
        unsafe_delete!(ir, dyn_maker)
    end


    ## PTX machine code

    @timeit to[] "LLVM back-end" begin
        @timeit to[] "preparation" prepare_execution!(job, ir)

        asm = @timeit to[] "machine-code generation" mcgen(job, ir, kernel)
    end

    target == :ptx && return asm, kernel_fn


    ## CUDA objects

    @timeit to[] "CUDA object generation" begin
        @timeit to[] "verification" begin
            check_invocation(job, kernel)
            check_ir(job, ir)
        end

        # enable debug options based on Julia's debug setting
        jit_options = Dict{CUDAdrv.CUjit_option,Any}()
        if Base.JLOptions().debug_level == 1
            jit_options[CUDAdrv.GENERATE_LINE_INFO] = true
        elseif Base.JLOptions().debug_level >= 2
            jit_options[CUDAdrv.GENERATE_DEBUG_INFO] = true
        end

        # link the CUDA device library
        @timeit to[] "linking" begin
            linker = CUDAdrv.CuLink(jit_options)
            CUDAdrv.add_file!(linker, libcudadevrt, CUDAdrv.LIBRARY)
            CUDAdrv.add_data!(linker, kernel_fn, asm)
            image = CUDAdrv.complete(linker)
        end

        @timeit to[] "compilation" begin
            cuda_mod = CuModule(image, jit_options)
            cuda_fun = CuFunction(cuda_mod, kernel_fn)
        end
    end

    target == :cuda && return cuda_fun, cuda_mod


    error("Unknown compilation target $target")
end
