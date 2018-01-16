# code reflection entry-points

# Return the capability of the current context's device, or a sane fall-back.
function current_capability()
    fallback = minimum(target_support)
    if !initialized[]
        return fallback
    end

    ctx = CuCurrentContext()
    if isnull(ctx)
        return fallback
    end

    return capability(device(ctx))
end


#
# code_* replacements
#

"""
    code_llvm([io], f, types; optimize=true, dump_module=false, cap::VersionNumber)

Prints the device LLVM IR generated for the method matching the given generic function and
type signature to `io` which defaults to `STDOUT`. The IR is optimized according to
`optimize` (defaults to true), and the entire module, including headers and other functions,
is dumped if `dump_module` is set (defaults to false). The device capability `cap` to
generate code for defaults to the current active device's capability, or v"2.0" if there is
no such active context.

See also: [`@device_code_llvm`](@ref), [`Base.code_llvm`](@ref)
"""
function code_llvm(io::IO, @nospecialize(func::Core.Function), @nospecialize(types=Tuple);
                   optimize::Bool=true, dump_module::Bool=false,
                   cap::VersionNumber=current_capability(), kernel::Bool=false)
    tt = Base.to_tuple_type(types)
    check_invocation(func, tt; kernel=kernel)

    mod, entry = irgen(func, tt)
    if kernel
        entry = wrap_entry!(mod, entry, tt)
    end
    if optimize
        optimize!(mod, entry, cap)
    end
    if dump_module
        show(io, mod)
    else
        show(io, entry)
    end
end
code_llvm(@nospecialize(func), @nospecialize(types=Tuple); kwargs...) = code_llvm(STDOUT, func, types; kwargs...)

"""
    code_ptx([io], f, types; cap::VersionNumber, kernel::Bool=false)

Prints the PTX assembly generated for the method matching the given generic function and
type signature to `io` which defaults to `STDOUT`. The device capability `cap` to generate
code for defaults to the current active device's capability, or v"2.0" if there is no such
active context. The optional `kernel` parameter indicates whether the function in question
is an entry-point function, or a regular device function.

See also: [`@device_code_ptx`](@ref)
"""
function code_ptx(io::IO, @nospecialize(func::Core.Function), @nospecialize(types=Tuple);
                  cap::VersionNumber=current_capability(), kernel::Bool=false)
    tt = Base.to_tuple_type(types)
    check_invocation(func, tt; kernel=kernel)

    ptx,_ = compile_function(func, tt, cap; kernel=kernel)
    # TODO: this code contains all the functions in the call chain,
    #       is it possible to implement `dump_module`?
    print(io, ptx)
end
code_ptx(@nospecialize(func), @nospecialize(types=Tuple); kwargs...) =
    code_ptx(STDOUT, func, types; kwargs...)

"""
    code_sass([io], f, types, cap::VersionNumber)

Prints the SASS code generated for the method matching the given generic function and type
signature to `io` which defaults to `STDOUT`. The device capability `cap` to generate code
for defaults to the current active device's capability, or v"2.0" if there is no such active
context. The method needs to be a valid entry-point kernel, eg. it should not return any
values.

See also: [`@device_code_sass`](@ref)
"""
function code_sass(io::IO, @nospecialize(func::Core.Function), @nospecialize(types=Tuple);
                   cap::VersionNumber=current_capability())
    tt = Base.to_tuple_type(types)
    check_invocation(func, tt; kernel=true)

    ptx,_ = compile_function(func, tt, cap)

    fn = tempname()
    gpu = "sm_$(cap.major)$(cap.minor)"
    # NOTE: this might not match what is being executed, due to the PTX->SASS conversion
    #       by the driver possibly not matching what `ptxas` (part of the toolkit) does.
    # TODO: see how `nvvp` extracts SASS code when doing PC sampling, and copy that.
    Base.run(`$ptxas --gpu-name $gpu --output-file $fn --input-as-string $ptx`)
    try
        print(io, read(`$cuobjdump --dump-sass $fn`, String))
    finally
        rm(fn)
    end
end
code_sass(@nospecialize(func), @nospecialize(types=Tuple); kwargs...) =
    code_sass(STDOUT, func, types; kwargs...)


#
# @device_code_* functions
#

export @device_code_lowered, @device_code_typed, @device_code_warntype,
       @device_code_llvm, @device_code_ptx, @device_code_sass

function emit_hooked_compilation(inner_hook, ex)
    quote
        # wipe the compile cache to force recompilation
        empty!(CUDAnative.compilecache)

        @assert CUDAnative.compile_hook[] == nothing
        try
            CUDAnative.compile_hook[] = $inner_hook
            $(esc(ex))
        finally
            CUDAnative.compile_hook[] = nothing
        end
    end
end

macro device_code_lowered(ex)
    # NOTE: these normally return values, so print instead
    hook = (func, tt, _) -> println(code_lowered(func, tt))
    emit_hooked_compilation(hook, ex)
end

macro device_code_typed(ex)
    # NOTE: these normally return values, so print instead
    hook = (func, tt, _) -> println(code_typed(func, tt))
    emit_hooked_compilation(hook, ex)
end

macro device_code_warntype(ex)
    hook = (func, tt, _) -> code_warntype(func, tt)
    emit_hooked_compilation(hook, ex)
end

macro device_code_llvm(ex)
    hook = (func, tt, cap) -> code_llvm(func, tt; cap=cap)
    emit_hooked_compilation(hook, ex)
end

macro device_code_ptx(ex)
    hook = (func, tt, cap) -> code_ptx(func, tt; cap=cap)
    emit_hooked_compilation(hook, ex)
end

macro device_code_sass(ex)
    hook = (func, tt, cap) -> code_sass(func, tt; cap=cap)
    emit_hooked_compilation(hook, ex)
end
