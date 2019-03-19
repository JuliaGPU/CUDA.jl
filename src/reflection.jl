# code reflection entry-points

using InteractiveUtils


#
# code_* replacements
#

# NOTE: these functions replicate parts of the main compiler driver in order to generate
#       more compact code (i.e. without the run-time library) and/or to support generating
#       otherwise invalid code (e.g. with missing symbols).

"""
    code_llvm([io], f, types; optimize=true, cap::VersionNumber, kernel=true,
                              dump_module=false, strip_ir_metadata=true)

Prints the device LLVM IR generated for the method matching the given generic function and
type signature to `io` which defaults to `stdout`. The IR is optimized according to
`optimize` (defaults to true), which includes entry-point specific optimizations if `kernel`
is set (defaults to false). The device capability `cap` to generate code for defaults to the
current active device's capability, or v"2.0" if there is no such active context. The entire
module, including headers and other functions, is dumped if `dump_module` is set (defaults
to false). Finally, setting `strip_ir_metadata` removes all debug metadata (defaults to
true).

See also: [`@device_code_llvm`](@ref), [`InteractiveUtils.code_llvm`](@ref)
"""
function code_llvm(io::IO, @nospecialize(func::Core.Function), @nospecialize(types);
                   optimize::Bool=true, cap::VersionNumber=current_capability(),
                   dump_module::Bool=false, strip_ir_metadata::Bool=true,
                   kernel::Bool=false, kwargs...)
    tt = Base.to_tuple_type(types)
    ctx = CompilerContext(func, tt, cap, kernel; kwargs...)
    code_llvm(io, ctx; optimize=optimize, dump_module=dump_module,
              strip_ir_metadata=strip_ir_metadata)
end
function code_llvm(io::IO, ctx::CompilerContext; optimize::Bool=true,
                   dump_module::Bool=false, strip_ir_metadata::Bool=true)
    ir, entry = compile(:llvm, ctx; optimize=optimize, strip=strip_ir_metadata)
    if dump_module
        show(io, ir)
    else
        show(io, entry)
    end
end
code_llvm(@nospecialize(func), @nospecialize(types); kwargs...) =
    code_llvm(stdout, func, types; kwargs...)

"""
    code_ptx([io], f, types; cap::VersionNumber, kernel=false, strip_ir_metadata=true)

Prints the PTX assembly generated for the method matching the given generic function and
type signature to `io` which defaults to `stdout`. The device capability `cap` to generate
code for defaults to the current active device's capability, or v"2.0" if there is no such
active context. The optional `kernel` parameter indicates whether the function in question
is an entry-point function, or a regular device function. Finally, setting
`strip_ir_metadata` removes all debug metadata (defaults to true).

See also: [`@device_code_ptx`](@ref)
"""
function code_ptx(io::IO, @nospecialize(func::Core.Function), @nospecialize(types);
                  cap::VersionNumber=current_capability(), kernel::Bool=false,
                  strip_ir_metadata::Bool=true, kwargs...)
    tt = Base.to_tuple_type(types)
    ctx = CompilerContext(func, tt, cap, kernel; kwargs...)
    code_ptx(io, ctx; strip_ir_metadata=strip_ir_metadata)
end
function code_ptx(io::IO, ctx::CompilerContext; strip_ir_metadata::Bool=true)
    asm, _ = compile(:ptx, ctx; strip=strip_ir_metadata)
    print(io, asm)
end
code_ptx(@nospecialize(func), @nospecialize(types); kwargs...) =
    code_ptx(stdout, func, types; kwargs...)

"""
    code_sass([io], f, types, cap::VersionNumber)

Prints the SASS code generated for the method matching the given generic function and type
signature to `io` which defaults to `stdout`. The device capability `cap` to generate code
for defaults to the current active device's capability, or v"2.0" if there is no such active
context. The method needs to be a valid entry-point kernel, eg. it should not return any
values.

See also: [`@device_code_sass`](@ref)
"""
function code_sass(io::IO, @nospecialize(func::Core.Function), @nospecialize(types);
                   cap::VersionNumber=current_capability(), kernel::Bool=true, kwargs...)
    tt = Base.to_tuple_type(types)
    ctx = CompilerContext(func, tt, cap, kernel; kwargs...)
    code_sass(io, ctx)
end
function code_sass(io::IO, ctx::CompilerContext)
    if !ctx.kernel
        error("Can only generate SASS code for kernel functions")
    end
    if ptxas === nothing || nvdisasm === nothing
        error("Your CUDA installation does not provide ptxas or nvdisasm, both of which are required for code_sass")
    end

    ptx, _ = compile(:ptx, ctx)

    fn = tempname()
    gpu = "sm_$(ctx.cap.major)$(ctx.cap.minor)"
    # NOTE: this might not match what is being executed, due to the PTX->SASS conversion
    #       by the driver possibly not matching what `ptxas` (part of the toolkit) does.
    # TODO: see how `nvvp` extracts SASS code when doing PC sampling, and copy that.
    Base.run(`$ptxas --gpu-name $gpu --output-file $fn --input-as-string $ptx`)
    try
        cmd = `$nvdisasm --print-code --print-line-info $fn`
        for line in readlines(cmd)
            # nvdisasm output is pretty verbose;
            # perform some clean-up and make it look like @code_native
            line = replace(line, r"/\*[0-9a-f]{4}\*/" => "        ") # strip inst addr
            line = replace(line, r"^[ ]{30}" => "   ")               # reduce leading spaces
            line = replace(line, r"[\s+]//##" => ";")                # change line info tag
            line = replace(line, r"^\." => "\n.")                    # break before new BBs
            line = replace(line, r"; File \"(.+?)\", line (\d+)" => s"; Location \1:\2") # rename line info
            println(io, line)
        end
    finally
        rm(fn)
    end
end
code_sass(@nospecialize(func), @nospecialize(types); kwargs...) =
    code_sass(stdout, func, types; kwargs...)


#
# @device_code_* functions
#

export @device_code_lowered, @device_code_typed, @device_code_warntype,
       @device_code_llvm, @device_code_ptx, @device_code_sass,
       @device_code

function emit_hooked_compilation(inner_hook, ex...)
    user_code = ex[end]
    user_kwargs = ex[1:end-1]
    quote
        # wipe the compile cache to force recompilation
        empty!(CUDAnative.compilecache)

        local kernels = 0
        function outer_hook(ctx)
            kernels += 1
            $inner_hook(ctx; $(map(esc, user_kwargs)...))
        end

        if CUDAnative.compile_hook[] != nothing
            error("Chaining multiple @device_code calls is unsupported")
        end
        try
            CUDAnative.compile_hook[] = outer_hook
            $(esc(user_code))
        finally
            CUDAnative.compile_hook[] = nothing
        end

        if kernels == 0
            error("no kernels executed while evaluating the given expression")
        end

        nothing
    end
end

# NOTE: these hooks take both a `f` and an inner `f`, because of how `@cuda`/`_cuda` work:
#       kernels are automatically wrapper in a function returning nothing, for usability.
#
#       Julia-level reflection (lowered/typed/warntype) skips these wrapper, because we
#       can't do call-site inlining and the kernel wrapper would hide any meaningful code.
#
#       at the LLVM level, we inline everything so there's no need to hide the wrapper.

"""
    @device_code_lowered ex

Evaluates the expression `ex` and returns the result of
[`InteractiveUtils.code_lowered`](@ref) for every compiled CUDA kernel.

See also: [`InteractiveUtils.@code_lowered`](@ref)
"""
macro device_code_lowered(ex...)
    quote
        buf = Any[]
        function hook(ctx::CompilerContext)
            append!(buf, code_lowered(ctx.f, ctx.tt))
        end
        $(emit_hooked_compilation(:hook, ex...))
        buf
    end
end

"""
    @device_code_typed ex

Evaluates the expression `ex` and returns the result of
[`InteractiveUtils.code_typed`](@ref) for every compiled CUDA kernel.

See also: [`InteractiveUtils.@code_typed`](@ref)
"""
macro device_code_typed(ex...)
    quote
        buf = Any[]
        function hook(ctx::CompilerContext)
            append!(buf, code_typed(ctx.f, ctx.tt))
        end
        $(emit_hooked_compilation(:hook, ex...))
        buf
    end
end

"""
    @device_code_warntype [io::IO=stdout] ex

Evaluates the expression `ex` and prints the result of
[`InteractiveUtils.code_warntype`](@ref) to `io` for every compiled CUDA kernel.

See also: [`InteractiveUtils.@code_warntype`](@ref)
"""
macro device_code_warntype(ex...)
    function hook(ctx::CompilerContext; io::IO=stdout, kwargs...)
        code_warntype(io, ctx.f, ctx.tt; kwargs...)
    end
    emit_hooked_compilation(hook, ex...)
end

"""
    @device_code_llvm [io::IO=stdout, ...] ex

Evaluates the expression `ex` and prints the result of [`InteractiveUtils.code_llvm`](@ref)
to `io` for every compiled CUDA kernel. For other supported keywords, see
[`CUDAnative.code_llvm`](@ref).

See also: [`InteractiveUtils.@code_llvm`](@ref)
"""
macro device_code_llvm(ex...)
    hook(ctx::CompilerContext; io::IO=stdout, kwargs...) = code_llvm(io, ctx; kwargs...)
    emit_hooked_compilation(hook, ex...)
end

"""
    @device_code_ptx [io::IO=stdout, ...] ex

Evaluates the expression `ex` and prints the result of [`CUDAnative.code_ptx`](@ref) to `io`
for every compiled CUDA kernel. For other supported keywords, see
[`CUDAnative.code_ptx`](@ref).
"""
macro device_code_ptx(ex...)
    hook(ctx::CompilerContext; io::IO=stdout, kwargs...) = code_ptx(io, ctx; kwargs...)
    emit_hooked_compilation(hook, ex...)
end

"""
    @device_code_sass [io::IO=stdout, ...] ex

Evaluates the expression `ex` and prints the result of [`CUDAnative.code_sass`](@ref) to
`io` for every compiled CUDA kernel. For other supported keywords, see
[`CUDAnative.code_sass`](@ref).
"""
macro device_code_sass(ex...)
    hook(ctx::CompilerContext; io::IO=stdout, kwargs...) = code_sass(io, ctx; kwargs...)
    emit_hooked_compilation(hook, ex...)
end

"""
    @device_code dir::AbstractString=... [...] ex

Evaluates the expression `ex` and dumps all intermediate forms of code to the directory
`dir`.
"""
macro device_code(ex...)
    only(xs) = (@assert length(xs) == 1; first(xs))
    function hook(ctx::CompilerContext; dir::AbstractString)
        fn = "$(typeof(ctx.f).name.mt.name)_$(globalUnique+1)"
        mkpath(dir)

        open(joinpath(dir, "$fn.lowered.jl"), "w") do io
            code = only(code_lowered(ctx.f, ctx.tt))
            println(io, code)
        end

        open(joinpath(dir, "$fn.typed.jl"), "w") do io
            code = only(code_typed(ctx.f, ctx.tt))
            println(io, code)
        end

        open(joinpath(dir, "$fn.unopt.ll"), "w") do io
            code_llvm(io, ctx; dump_module=true, strip_ir_metadata=false, optimize=false)
        end

        open(joinpath(dir, "$fn.opt.ll"), "w") do io
            code_llvm(io, ctx; dump_module=true, strip_ir_metadata=false)
        end

        open(joinpath(dir, "$fn.ptx"), "w") do io
            code_ptx(io, ctx)
        end

        open(joinpath(dir, "$fn.sass"), "w") do io
            code_sass(io, ctx)
        end
    end
    emit_hooked_compilation(hook, ex...)
end
