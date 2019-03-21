# compiler driver and main interface

# (::CompilerJob)
const compile_hook = Ref{Union{Nothing,Function}}(nothing)

"""
    compile(to::Symbol, cap::VersionNumber, f, tt, kernel=true;
            kernel=true, optimize=true, strip=false, ...)

Compile a function `f` invoked with types `tt` for device capability `cap` to one of the
following formats as specified by the `to` argument: `:julia` for Julia IR, `:llvm` for LLVM
IR, `:ptx` for PTX assembly and `:cuda` for CUDA driver objects. If the `kernel` flag is
set, specialized code generation and optimization for kernel functions is enabled.

The following keyword arguments are supported:
- `hooks`: enable compiler hooks that drive reflection functions (default: true)
- `libraries`: link auxiliary bitcode libraries that may be required (default: true)
- `optimize`: optimize the code (default: true)
- `strip`: strip non-functional metadata and debug information  (default: false)

Other keyword arguments can be found in the documentation of [`cufunction`](@ref).
"""
compile(to::Symbol, cap::VersionNumber, @nospecialize(f::Core.Function), @nospecialize(tt),
        kernel::Bool=true; hooks::Bool=true, libraries::Bool=true,
        optimize::Bool=true, strip::Bool=false,
        kwargs...) =
    compile(to, CompilerJob(f, tt, cap, kernel; kwargs...);
            hooks=hooks, libraries=libraries, optimize=optimize, strip=strip)

function compile(to::Symbol, job::CompilerJob;
                 hooks::Bool=true, libraries::Bool=true,
                 optimize::Bool=true, strip::Bool=false)
    @debug "(Re)compiling function" job

    if hooks && compile_hook[] != nothing
        global globalUnique
        previous_globalUnique = globalUnique

        compile_hook[](job)

        globalUnique = previous_globalUnique
    end


    ## Julia IR

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
    linfo = ccall(:jl_specializations_get_linfo, Ref{Core.MethodInstance},
                  (Any, Any, Any, UInt), meth, ti, env, world)

    to == :julia && return linfo


    ## LLVM IR

    ir, kernel = irgen(job, linfo, world)

    need_library(lib) = any(f -> isdeclaration(f) &&
                                 intrinsic_id(f) == 0 &&
                                 haskey(functions(lib), LLVM.name(f)),
                            functions(ir))

    if libraries
        libdevice = load_libdevice(job.cap)
        if need_library(libdevice)
            link_libdevice!(job, ir, libdevice)
        end
    end

    if optimize
        kernel = optimize!(job, ir, kernel)
    end

    if libraries
        runtime = load_runtime(job.cap)
        if need_library(runtime)
            link_library!(job, ir, runtime)
        end
    end

    verify(ir)

    if strip
        strip_debuginfo!(ir)
    end

    kernel_fn = LLVM.name(kernel)
    kernel_ft = eltype(llvmtype(kernel))

    to == :llvm && return ir, kernel


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
                ops = collect(operands(call))[1:2]
                ## addrspacecast
                ops = LLVM.Value[first(operands(val)) for val in ops]
                ## inttoptr
                ops = ConstantInt[first(operands(val)) for val in ops]
                ## integer constants
                ops = convert.(Int, ops)
                ## actual pointer values
                ops = Ptr{Any}.(ops)

                dyn_f, dyn_tt = unsafe_pointer_to_objref.(ops)
                dyn_job = CompilerJob(dyn_f, dyn_tt, job.cap, #=kernel=# true)
                push!(worklist, dyn_job => call)
            end

            # compile and link
            for dyn_job in keys(worklist)
                # cached compilation
                dyn_kernel_fn = get!(kernels, dyn_job) do
                    dyn_ir, dyn_kernel = compile(:llvm, dyn_job; hooks=false,
                                                 optimize=optimize, strip=strip)
                    dyn_kernel_fn = LLVM.name(dyn_kernel)
                    dyn_kernel_ft = eltype(llvmtype(dyn_kernel))
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

    prepare_execution!(job, ir)

    check_invocation(job, kernel)
    check_ir(job, ir)

    asm = mcgen(job, ir, kernel)

    to == :ptx && return asm, kernel_fn


    ## CUDA objects

    # enable debug options based on Julia's debug setting
    jit_options = Dict{CUDAdrv.CUjit_option,Any}()
    if Base.JLOptions().debug_level == 1
        jit_options[CUDAdrv.GENERATE_LINE_INFO] = true
    elseif Base.JLOptions().debug_level >= 2
        jit_options[CUDAdrv.GENERATE_DEBUG_INFO] = true
    end

    # link the CUDA device library
    linker = CUDAdrv.CuLink(jit_options)
    CUDAdrv.add_file!(linker, libcudadevrt, CUDAdrv.LIBRARY)
    CUDAdrv.add_data!(linker, kernel_fn, asm)
    image = CUDAdrv.complete(linker)

    cuda_mod = CuModule(image, jit_options)
    cuda_fun = CuFunction(cuda_mod, kernel_fn)

    to == :cuda && return cuda_fun, cuda_mod


    error("Unknown compilation target $to")
end
