# compiler driver and main interface

# (::CompilerContext)
const compile_hook = Ref{Union{Nothing,Function}}(nothing)

"""
    compile(cap::VersionNumber, f, tt; kernel=true, kwargs...)

Compile a function `f` invoked with types `tt` for device `dev` or its compute capability
`cap`, returning the compiled function module respectively of type `CuFuction` and
`CuModule`.

For a list of supported keyword arguments, refer to the documentation of
[`cufunction`](@ref).
"""
compile(cap::VersionNumber, @nospecialize(f::Core.Function), @nospecialize(tt);
        kernel=true, kwargs...) =
    compile(CompilerContext(f, tt, cap, kernel; kwargs...))

function compile(ctx::CompilerContext)
    CUDAnative.configured || error("CUDAnative.jl has not been configured; cannot JIT code.")

    # generate code
    ir, entry = codegen(ctx)
    check_invocation(ctx, entry)
    check_ir(ctx, ir)
    verify(ir)
    asm = mcgen(ctx, ir, entry)

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

    return cuda_fun, cuda_mod
end

codegen(cap::VersionNumber, @nospecialize(f::Core.Function), @nospecialize(tt);
        kernel=true, kwargs...) =
    codegen(CompilerContext(f, tt, cap, kernel; kwargs...))

function codegen(ctx::CompilerContext)
    if compile_hook[] != nothing
        hook = compile_hook[]
        compile_hook[] = nothing

        global globalUnique
        previous_globalUnique = globalUnique

        hook(ctx)

        globalUnique = previous_globalUnique
        compile_hook[] = hook
    end


    ## high-level code generation (Julia AST)

    @debug "(Re)compiling function" ctx

    check_method(ctx)


    ## low-level code generation (LLVM IR)

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
    entry = optimize!(ctx, ir, entry)

    runtime = load_runtime(ctx.cap)
    if need_library(runtime)
        link_library!(ctx, ir, runtime)
    end

    prepare_execution!(ctx, ir)


    ## dynamic parallelism

    # find dynamic kernel invocations
    dyn_calls = []
    if haskey(functions(ir), "cudanativeLaunchDevice")
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
    end

    # compile and link
    for (call, dyn_f, dyn_tt) in dyn_calls
        dyn_ctx = CompilerContext(dyn_f, dyn_tt, ctx.cap, true)
        dyn_ir, dyn_entry = codegen(dyn_ctx)
        link_library!(ctx, ir, dyn_ir)

        # TODO
        unsafe_delete!(LLVM.parent(call), call)
    end


    ## finalization

    return ir, entry
end
