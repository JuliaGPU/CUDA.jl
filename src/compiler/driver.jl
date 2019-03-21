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
- `optimize`: optimize the code (default: true)
- `strip`: strip non-functional metadata and debug information  (default: false)

Other keyword arguments can be found in the documentation of [`cufunction`](@ref).
"""
compile(to::Symbol, cap::VersionNumber, @nospecialize(f::Core.Function), @nospecialize(tt),
        kernel::Bool=true; hooks::Bool=true, optimize::Bool=true, strip::Bool=false,
        kwargs...) =
    compile(to, CompilerJob(f, tt, cap, kernel; kwargs...);
            hooks=hooks, optimize=optimize, strip=strip)

function compile(to::Symbol, job::CompilerJob;
                 hooks::Bool=true, optimize::Bool=true, strip::Bool=false)
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

    ir, entry = irgen(job, linfo, world)

    need_library(lib) = any(f -> isdeclaration(f) &&
                                 intrinsic_id(f) == 0 &&
                                 haskey(functions(lib), LLVM.name(f)),
                            functions(ir))

    libdevice = load_libdevice(job.cap)
    if need_library(libdevice)
        link_libdevice!(job, ir, libdevice)
    end

    # optimize the IR
    if optimize
        entry = optimize!(job, ir, entry)
    end

    runtime = load_runtime(job.cap)
    if need_library(runtime)
        link_library!(job, ir, runtime)
    end

    verify(ir)

    if strip
        strip_debuginfo!(ir)
    end


    ## dynamic parallelism

    if haskey(functions(ir), "cudanativeCompileKernel")
        f = functions(ir)["cudanativeCompileKernel"]

        # find dynamic kernel invocations
        # TODO: recover this information earlier, from the Julia IR
        worklist = Dict{Tuple{Core.Function,Type}, Vector{LLVM.CallInst}}()
        for use in uses(f)
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
            calls = get!(worklist, (dyn_f, dyn_tt), LLVM.CallInst[])
            push!(calls, call)
        end

        # compile and link
        for (dyn_f, dyn_tt) in keys(worklist)
            dyn_ctx = CompilerJob(dyn_f, dyn_tt, job.cap, true)
            dyn_ir, dyn_entry =
                compile(:llvm, dyn_ctx; hooks=false, optimize=optimize, strip=strip)

            dyn_fn = LLVM.name(dyn_entry)
            link!(ir, dyn_ir)
            dyn_ir = nothing
            dyn_entry = functions(ir)[dyn_fn]

            # insert a call everywhere the kernel is used
            for call in worklist[(dyn_f,dyn_tt)]
                replace_uses!(call, dyn_entry)
                unsafe_delete!(LLVM.parent(call), call)
            end
        end

        @compiler_assert isempty(uses(f)) job
        unsafe_delete!(ir, f)
    end

    to == :llvm && return ir, entry


    ## PTX machine code

    prepare_execution!(job, ir)

    check_invocation(job, entry)
    check_ir(job, ir)

    asm = mcgen(job, ir, entry)

    to == :ptx && return asm, LLVM.name(entry)


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
    CUDAdrv.add_data!(linker, LLVM.name(entry), asm)
    image = CUDAdrv.complete(linker)

    cuda_mod = CuModule(image, jit_options)
    cuda_fun = CuFunction(cuda_mod, LLVM.name(entry))

    to == :cuda && return cuda_fun, cuda_mod


    error("Unknown compilation target $to")
end
