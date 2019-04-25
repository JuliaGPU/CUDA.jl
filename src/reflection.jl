# code reflection entry-points

using InteractiveUtils


#
# code_* replacements
#

# NOTE: these functions replicate parts of the main compiler driver in order to generate
#       more compact code (i.e. without the run-time library) and/or to support generating
#       otherwise invalid code (e.g. with missing symbols).

"""
    code_llvm([io], f, types; optimize=true, cap::VersionNumber, kernel=false,
              optimize=true, raw=false, dump_module=false, strict=false)

Prints the device LLVM IR generated for the method matching the given generic function and
type signature to `io` which defaults to `stdout`.

The following keyword arguments are supported:

- `cap` which device to generate code for
- `kernel`: treat the function as an entry-point kernel
- `optimize`: determines if the code is optimized, which includes kernel-specific
  optimizations if `kernel` is true
- `raw`: return the raw IR including all metadata
- `dump_module`: display the entire module instead of just the function
- `strict`: verify generate code as early as possible

See also: [`@device_code_llvm`](@ref), [`InteractiveUtils.code_llvm`](@ref)
"""
function code_llvm(io::IO, @nospecialize(func), @nospecialize(types);
                   cap::VersionNumber=current_capability(), kernel::Bool=false,
                   optimize::Bool=true, raw::Bool=false,
                   dump_module::Bool=false, strict::Bool=false, kwargs...)
    tt = Base.to_tuple_type(types)
    job = CompilerJob(func, tt, cap, kernel; kwargs...)
    code_llvm(io, job; optimize=optimize,
              raw=raw, dump_module=dump_module, strict=strict)
end
function code_llvm(io::IO, job::CompilerJob; optimize::Bool=true, raw::Bool=false,
                   dump_module::Bool=false, strict::Bool=false)
    ir, entry = codegen(:llvm, job; optimize=optimize, strip=!raw, strict=strict)
    if dump_module
        show(io, ir)
    else
        show(io, entry)
    end
end
code_llvm(@nospecialize(func), @nospecialize(types); kwargs...) =
    code_llvm(stdout, func, types; kwargs...)

"""
    code_ptx([io], f, types; cap::VersionNumber, kernel=false, raw=false, strict=false)

Prints the PTX assembly generated for the method matching the given generic function and
type signature to `io` which defaults to `stdout`.

The following keyword arguments are supported:

- `cap` which device to generate code for
- `kernel`: treat the function as an entry-point kernel
- `raw`: return the raw code including all metadata
- `strict`: verify generate code as early as possible

See also: [`@device_code_ptx`](@ref)
"""
function code_ptx(io::IO, @nospecialize(func), @nospecialize(types);
                  cap::VersionNumber=current_capability(), kernel::Bool=false,
                  raw::Bool=false, strict::Bool=false, kwargs...)
    tt = Base.to_tuple_type(types)
    job = CompilerJob(func, tt, cap, kernel; kwargs...)
    code_ptx(io, job; raw=raw, strict=strict)
end
function code_ptx(io::IO, job::CompilerJob; raw::Bool=false, strict::Bool=false)
    asm, _ = codegen(:ptx, job; strip=!raw, strict=strict)
    print(io, asm)
end
code_ptx(@nospecialize(func), @nospecialize(types); kwargs...) =
    code_ptx(stdout, func, types; kwargs...)

"""
    code_sass([io], f, types, cap::VersionNumber)

Prints the SASS code generated for the method matching the given generic function and type
signature to `io` which defaults to `stdout`.

The following keyword arguments are supported:

- `cap` which device to generate code for
- `kernel`: treat the function as an entry-point kernel

See also: [`@device_code_sass`](@ref)
"""
function code_sass(io::IO, @nospecialize(func), @nospecialize(types);
                   cap::VersionNumber=current_capability(), kernel::Bool=true, kwargs...)
    tt = Base.to_tuple_type(types)
    job = CompilerJob(func, tt, cap, kernel; kwargs...)
    code_sass(io, job)
end
function code_sass(io::IO, job::CompilerJob)
    if !job.kernel
        error("Can only generate SASS code for kernel functions")
    end
    if ptxas === nothing || nvdisasm === nothing
        error("Your CUDA installation does not provide ptxas or nvdisasm, both of which are required for code_sass")
    end

    ptx, _ = codegen(:ptx, job)

    fn = tempname()
    gpu = "sm_$(job.cap.major)$(job.cap.minor)"
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
        function outer_hook(job)
            kernels += 1
            $inner_hook(job; $(map(esc, user_kwargs)...))
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
        function hook(job::CompilerJob)
            append!(buf, code_lowered(job.f, job.tt))
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
        function hook(job::CompilerJob)
            append!(buf, code_typed(job.f, job.tt))
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
    function hook(job::CompilerJob; io::IO=stdout, kwargs...)
        code_warntype(io, job.f, job.tt; kwargs...)
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
    hook(job::CompilerJob; io::IO=stdout, kwargs...) = code_llvm(io, job; kwargs...)
    emit_hooked_compilation(hook, ex...)
end

"""
    @device_code_ptx [io::IO=stdout, ...] ex

Evaluates the expression `ex` and prints the result of [`CUDAnative.code_ptx`](@ref) to `io`
for every compiled CUDA kernel. For other supported keywords, see
[`CUDAnative.code_ptx`](@ref).
"""
macro device_code_ptx(ex...)
    hook(job::CompilerJob; io::IO=stdout, kwargs...) = code_ptx(io, job; kwargs...)
    emit_hooked_compilation(hook, ex...)
end

"""
    @device_code_sass [io::IO=stdout, ...] ex

Evaluates the expression `ex` and prints the result of [`CUDAnative.code_sass`](@ref) to
`io` for every compiled CUDA kernel. For other supported keywords, see
[`CUDAnative.code_sass`](@ref).
"""
macro device_code_sass(ex...)
    hook(job::CompilerJob; io::IO=stdout, kwargs...) = code_sass(io, job; kwargs...)
    emit_hooked_compilation(hook, ex...)
end

"""
    @device_code dir::AbstractString=... [...] ex

Evaluates the expression `ex` and dumps all intermediate forms of code to the directory
`dir`.
"""
macro device_code(ex...)
    only(xs) = (@assert length(xs) == 1; first(xs))
    function hook(job::CompilerJob; dir::AbstractString)
        fn = "$(typeof(job.f).name.mt.name)_$(globalUnique+1)"
        mkpath(dir)

        open(joinpath(dir, "$fn.lowered.jl"), "w") do io
            code = only(code_lowered(job.f, job.tt))
            println(io, code)
        end

        open(joinpath(dir, "$fn.typed.jl"), "w") do io
            code = only(code_typed(job.f, job.tt))
            println(io, code)
        end

        open(joinpath(dir, "$fn.unopt.ll"), "w") do io
            code_llvm(io, job; dump_module=true, raw=true, optimize=false)
        end

        open(joinpath(dir, "$fn.opt.ll"), "w") do io
            code_llvm(io, job; dump_module=true, raw=true)
        end

        open(joinpath(dir, "$fn.ptx"), "w") do io
            code_ptx(io, job)
        end

        open(joinpath(dir, "$fn.sass"), "w") do io
            code_sass(io, job)
        end
    end
    emit_hooked_compilation(hook, ex...)
end
