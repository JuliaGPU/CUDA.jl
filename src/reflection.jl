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

function emit_hooked_compilation(inner_hook, ex...)
    user_code = ex[end]
    kwargs = ex[1:end-1]
    quote
        # wipe the compile cache to force recompilation
        empty!(CUDAnative.compilecache)

        local kernels = 0
        function outer_hook(args...)
            kernels += 1
            $inner_hook(args...; $(map(esc, kwargs)...))
        end

        @assert CUDAnative.compile_hook[] == nothing
        try
            CUDAnative.compile_hook[] = outer_hook
            $(esc(user_code))
        catch ex
            warn(ex)
        finally
            CUDAnative.compile_hook[] = nothing
        end

        if kernels == 0
            error("no kernels executed while evaluating the given expression")
        end

        nothing
    end
end

"""
    @device_code_lowered ex

Evaluates the expression `ex` and returns the result of [`Base.code_lowered`](@ref) for
every compiled CUDA kernel.

See also: [`Base.@code_lowered`](@ref)
"""
macro device_code_lowered(ex...)
    @gensym hook
    quote
        buf = Any[]
        function $hook(func, tt, cap)
            append!(buf, code_lowered(func, tt))
        end
        $(emit_hooked_compilation(hook, ex...))
        buf
    end
end

"""
    @device_code_typed ex

Evaluates the expression `ex` and returns the result of [`Base.code_typed`](@ref) for
every compiled CUDA kernel.

See also: [`Base.@code_typed`](@ref)
"""
macro device_code_typed(ex...)
    @gensym hook
    quote
        buf = Any[]
        function $hook(func, tt, cap)
            append!(buf, code_typed(func, tt))
        end
        $(emit_hooked_compilation(hook, ex...))
        buf
    end
end

"""
    @device_code_warntype [io::IO=STDOUT] ex

Evaluates the expression `ex` and prints the result of [`Base.code_warntype`](@ref) to `io`
for every compiled CUDA kernel.

See also: [`Base.@code_warntype`](@ref)
"""
macro device_code_warntype(ex...)
    function hook(func, tt, cap; io::IO=STDOUT)
        code_warntype(io, func, tt)
    end
    emit_hooked_compilation(hook, ex...)
end

"""
    @device_code_llvm [io::IO=STDOUT] [optimize::Bool=true] [dump_module::Bool=false] ex

Evaluates the expression `ex` and prints the result of [`Base.code_llvm`](@ref) to `io` for
every compiled CUDA kernel. The `optimize` keyword argument determines whether the code is
optimized, and `dump_module` can be used to print the entire LLVM module instead of only the
entry-point function.

See also: [`Base.@code_llvm`](@ref)
"""
macro device_code_llvm(ex...)
    function hook(func, tt, cap; io::IO=STDOUT, optimize::Bool=true, dump_module::Bool=false)
        code_llvm(io, func, tt; kernel=true, cap=cap, optimize=optimize, dump_module=dump_module)
    end
    emit_hooked_compilation(hook, ex...)
end

"""
    @device_code_ptx [io::IO=STDOUT] ex

Evaluates the expression `ex` and prints the result of [`CUDAnative.code_ptx`](@ref) to `io`
for every compiled CUDA kernel.
"""
macro device_code_ptx(ex...)
    function hook(func, tt, cap; io::IO=STDOUT)
        code_ptx(io, func, tt; kernel=true, cap=cap)
    end
    emit_hooked_compilation(hook, ex...)
end

"""
    @device_code_sass [io::IO=STDOUT] ex

Evaluates the expression `ex` and prints the result of [`CUDAnative.code_sass`](@ref) to
`io` for every compiled CUDA kernel.
"""
macro device_code_sass(ex...)
    function hook(func, tt, cap; io::IO=STDOUT)
        code_sass(io, func, tt; cap=cap)
    end
    emit_hooked_compilation(hook, ex...)
end
