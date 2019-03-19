# compiler driver and main interface

# (::CompilerContext)
const compile_hook = Ref{Union{Nothing,Function}}(nothing)

"""
    compile(to::Symbol, cap::VersionNumber, f, tt;
            kernel=true, optimize=true, strip=false, ...)

Compile a function `f` invoked with types `tt` for device capability `cap` to one of the
following formats as specified by the `to` argument: `:julia` for Julia IR, `:llvm` for LLVM
IR, `:ptx` for PTX assembly and `:cuda` for CUDA driver objects.

The following keyword arguments are supported:
- `kernel`: enable kernel-specific code generation
- `optimize`: optimize the code
- `strip`: strip non-functional metadata and debug information

Other keyword arguments can be found in the documentation of [`cufunction`](@ref).
"""
compile(to::Symbol, cap::VersionNumber, @nospecialize(f::Core.Function), @nospecialize(tt);
        kernel::Bool=true, optimize::Bool=true, strip::Bool=false, kwargs...) =
    compile(to, CompilerContext(f, tt, cap, kernel; kwargs...);
            optimize=optimize, strip=strip)

function compile(to::Symbol, ctx::CompilerContext;
                 optimize::Bool=true, strip::Bool=false)
    @debug "(Re)compiling function" ctx

    if compile_hook[] != nothing
        hook = compile_hook[]
        compile_hook[] = nothing

        global globalUnique
        previous_globalUnique = globalUnique

        hook(ctx)

        globalUnique = previous_globalUnique
        compile_hook[] = hook
    end


    ## Julia IR

    check_method(ctx)

    # TODO: get the method here, don't put it in the context?
    #to == :julia && return asm


    ## LLVM IR

    ir, entry = irgen(ctx)

    need_library(lib) = any(f -> isdeclaration(f) &&
                                 intrinsic_id(f) == 0 &&
                                 haskey(functions(lib), LLVM.name(f)),
                            functions(ir))

    libdevice = load_libdevice(ctx.cap)
    if need_library(libdevice)
        link_libdevice!(ctx, ir, libdevice)
    end

    # optimize the IR
    if optimize
        entry = optimize!(ctx, ir, entry)
    end

    runtime = load_runtime(ctx.cap)
    if need_library(runtime)
        link_library!(ctx, ir, runtime)
    end

    verify(ir)

    if strip
        strip_debuginfo!(ir)
    end


    ## dynamic parallelism

    if haskey(functions(ir), "cudanativeLaunchDevice")
        dyn_calls = []

        # find dynamic kernel invocations
        f = functions(ir)["cudanativeLaunchDevice"]
        for use in uses(f)
            # decode the call
            # FIXME: recover this earlier, from the Julia IR
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
            push!(dyn_calls, (call, dyn_f, dyn_tt))
        end

        # compile and link
        for (call, dyn_f, dyn_tt) in dyn_calls
            # disable the compile hook; this recursive compilation call
            # shouldn't be traced separately
            hook = compile_hook[]
            compile_hook[] = nothing

            dyn_ctx = CompilerContext(dyn_f, dyn_tt, ctx.cap, true)
            dyn_ir, dyn_entry =
                compile(:llvm, dyn_ctx; optimize=optimize, strip=strip)

            compile_hook[] = hook

            link!(ir, dyn_ir)

            # TODO
            unsafe_delete!(LLVM.parent(call), call)
        end

        @compiler_assert isempty(uses(f)) ctx
        unsafe_delete!(ir, f)
    end

    to == :llvm && return ir, entry


    ## PTX machine code

    prepare_execution!(ctx, ir)

    check_invocation(ctx, entry)
    check_ir(ctx, ir)

    asm = mcgen(ctx, ir, entry)

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
