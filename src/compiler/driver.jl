# compiler driver and main interface

# (::CompilerContext)
const compile_hook = Ref{Union{Nothing,Function}}(nothing)

"""
    compile(dev::CuDevice, f, tt; kwargs...)

Compile a function `f` invoked with types `tt` for device `dev`, returning the compiled
function module respectively of type `CuFuction` and `CuModule`.

For a list of supported keyword arguments, refer to the documentation of
[`cufunction`](@ref).
"""
function compile(dev::CuDevice, @nospecialize(f::Core.Function), @nospecialize(tt); kwargs...)
    CUDAnative.configured || error("CUDAnative.jl has not been configured; cannot JIT code.")

    module_asm, module_entry = compile(supported_capability(dev), f, tt; kwargs...)

    # enable debug options based on Julia's debug setting
    jit_options = Dict{CUDAdrv.CUjit_option,Any}()
    if Base.JLOptions().debug_level == 1
        jit_options[CUDAdrv.GENERATE_LINE_INFO] = true
    elseif Base.JLOptions().debug_level >= 2
        jit_options[CUDAdrv.GENERATE_DEBUG_INFO] = true
    end
    cuda_mod = CuModule(module_asm, jit_options)
    cuda_fun = CuFunction(cuda_mod, module_entry)

    return cuda_fun, cuda_mod
end

# same as above, but without an active device
function compile(cap::VersionNumber, @nospecialize(f), @nospecialize(tt);
                 kernel=true, kwargs...)
    ctx = CompilerContext(f, tt, cap, kernel; kwargs...)

    return compile(ctx)
end

function compile(ctx::CompilerContext)
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

    mod, entry = irgen(ctx)

    need_library(lib) = any(f -> isdeclaration(f) &&
                                 intrinsic_id(f) == 0 &&
                                 haskey(functions(lib), LLVM.name(f)),
                            functions(mod))

    libdevice = load_libdevice(ctx.cap)
    if need_library(libdevice)
        link_libdevice!(ctx, mod, libdevice)
    end

    # optimize the IR
    entry = optimize!(ctx, mod, entry)

    runtime = load_runtime(ctx.cap)
    if need_library(runtime)
        link_library!(ctx, mod, runtime)
    end

    prepare_execution!(ctx, mod)

    check_invocation(ctx, entry)

    # check generated IR
    check_ir(ctx, mod)
    verify(mod)


    ## machine code generation (PTX assembly)

    module_asm = mcgen(ctx, mod, entry)

    return module_asm, LLVM.name(entry)
end
