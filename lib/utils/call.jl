# utilities for calling foreign functionality more conveniently

export @checked, with_workspace, with_workspaces,
       @debug_ccall, @gcsafe_ccall, @gcunsafe_callback


## function wrapper for checking the return value of a function

"""
    @checked function foo(...)
        rv = ...
        return rv
    end

Macro for wrapping a function definition returning a status code. Two versions of the
function will be generated: `foo`, with the function execution wrapped by an invocation of
the `check` function (to be implemented by the caller of this macro), and `unchecked_foo`
where no such invocation is present and the status code is returned to the caller.
"""
macro checked(ex)
    # parse the function definition
    @assert Meta.isexpr(ex, :function)
    sig = ex.args[1]
    @assert Meta.isexpr(sig, :call)
    body = ex.args[2]
    @assert Meta.isexpr(body, :block)

    # generate a "safe" version that performs a check
    safe_body = quote
        check() do
            $body
        end
    end
    safe_sig = Expr(:call, sig.args[1], sig.args[2:end]...)
    safe_def = Expr(:function, safe_sig, safe_body)

    # generate a "unchecked" version that returns the error code instead
    unchecked_sig = Expr(:call, Symbol("unchecked_", sig.args[1]), sig.args[2:end]...)
    unchecked_def = Expr(:function, unchecked_sig, body)

    return esc(:($safe_def, $unchecked_def))
end


## wrapper for foreign functionality that requires a workspace buffer

"""
    with_workspace([eltyp=UInt8], size, [fallback::Int]; keep::Bool=false) do workspace
        ...
    end

Create a GPU workspace vector with element type `eltyp` and size in number of elements (in
the default case of an UInt8 element type this equals to the amount of bytes) specified by
`size`, and pass it to the do block. A fallback workspace size `fallback` can be specified
if the regular size would lead to OOM. Afterwards, the buffer is put back into the memory
pool for reuse (unless `keep` is set to `true`).

This helper protects against the rare but real issue of the workspace size getter returning
different results based on the GPU device memory pressure, which might change _after_
initial allocation of the workspace (which can cause a GC collection).
"""
@inline with_workspace(f::Base.Callable, size::Union{Integer,Function}, fallback::Union{Nothing,Integer}=nothing; keep::Bool=false) =
    with_workspaces(f, UInt8, size, -1, fallback; keep)

@inline with_workspace(f::Base.Callable, eltyp::Type{T}, size::Union{Integer,Function},
                       fallback::Union{Nothing,Integer}=nothing; keep::Bool=false) where {T} =
    with_workspaces(f, T, size, -1, fallback; keep)

"""
    with_workspaces([eltyp=UInt8], size_gpu, size_cpu, [fallback::Int]; keep::Bool=false) do workspace_gpu, workspace_cpu
        ...
    end

Create GPU and CPU workspace vectors with element type `eltyp` and size in number of elements (in
the default case of an UInt8 element type this equals to the amount of bytes) specified by
`size_gpu` and `size_cpu`, and pass them to the do block.
A fallback GPU workspace size `fallback` can be specified if the regular size would lead to OOM.
Afterwards, the GPU buffer is put back into the memory pool for reuse (unless `keep` is set to `true`).
This helper protects against the rare but real issue of the GPU workspace size getter returning
different results based on the GPU device memory pressure, which might change _after_
initial allocation of the workspace (which can cause a GC collection).
"""
@inline with_workspaces(f::Base.Callable, size_gpu::Union{Integer,Function}, size_cpu::Union{Integer,Function},
                        fallback::Union{Nothing,Integer}=nothing; keep::Bool=false) =
    with_workspaces(f, UInt8, size_gpu, size_cpu, fallback; keep)

function with_workspaces(f::Base.Callable, eltyp::Type{T}, size_gpu::Union{Integer,Function}, size_cpu::Union{Integer,Function},
                         fallback::Union{Nothing,Integer}=nothing; keep::Bool=false) where {T}

    get_size_gpu() = Int(isa(size_gpu, Integer) ? size_gpu : size_gpu()::Integer)
    get_size_cpu() = Int(isa(size_cpu, Integer) ? size_cpu : size_cpu()::Integer)

    sz_gpu = get_size_gpu()
    sz_cpu = get_size_cpu()

    workspace_gpu = nothing
    try
        while workspace_gpu === nothing || length(workspace_gpu) < sz_gpu
            workspace_gpu = CuVector{T}(undef, sz_gpu)
            sz_gpu = get_size_gpu()
        end
    catch ex
        fallback === nothing && rethrow()
        isa(ex, OutOfGPUMemoryError) || rethrow()
        workspace_gpu = CuVector{T}(undef, fallback)
    end
    workspace_gpu = workspace_gpu::CuVector{T}

    # use & free
    try
        # size_cpu == -1 means that we don't need a CPU workspace
        if sz_cpu == -1
            f(workspace_gpu)
        else
            workspace_cpu = Vector{T}(undef, sz_cpu)
            f(workspace_gpu, workspace_cpu)
        end
    finally
        keep || CUDA.unsafe_free!(workspace_gpu)
    end
end


## version of ccall that prints the ccall, its arguments and its return value

macro debug_ccall(ex)
    @assert Meta.isexpr(ex, :(::))
    call, ret = ex.args
    @assert Meta.isexpr(call, :call)
    target, argexprs... = call.args
    args = map(argexprs) do argexpr
        @assert Meta.isexpr(argexpr, :(::))
        argexpr.args[1]
    end

    ex = Expr(:macrocall, Symbol("@ccall"), __source__, ex)

    # avoid task switches
    io = :(Core.stdout)

    quote
        print($io, $(string(target)), '(')
        for (i, arg) in enumerate(($(map(esc, args)...),))
            i > 1 && print($io, ", ")
            render_arg($io, arg)
        end
        print($io, ')')

        rv = $(esc(ex))

        println($io, " = ", rv)
        for (i, arg) in enumerate(($(map(esc, args)...),))
            if arg isa Base.RefValue
                println($io, " $i: ", arg[])
            end
        end
        rv
    end
end

render_arg(io, arg) = print(io, arg)
render_arg(io, arg::Union{<:Base.RefValue, AbstractArray}) = summary(io, arg)


## version of ccall that calls jl_gc_safe_enter|leave around the inner ccall

# TODO: replace with JuliaLang/julia#49933 once merged

# note that this is generally only safe with functions that do not call back into Julia.
# when callbacks occur, the code should ensure the GC is not running by wrapping the code
# in the `@gcunsafe` macro

function ccall_macro_lower(func, rettype, types, args, nreq)
    # instead of re-using ccall or Expr(:foreigncall) to perform argument conversion,
    # we need to do so ourselves in order to insert a jl_gc_safe_enter|leave
    # just around the inner ccall

    cconvert_exprs = []
    cconvert_args = []
    for (typ, arg) in zip(types, args)
        var = gensym("$(func)_cconvert")
        push!(cconvert_args, var)
        push!(cconvert_exprs, quote
            $var = Base.cconvert($(esc(typ)), $(esc(arg)))
        end)
    end

    unsafe_convert_exprs = []
    unsafe_convert_args = []
    for (typ, arg) in zip(types, cconvert_args)
        var = gensym("$(func)_unsafe_convert")
        push!(unsafe_convert_args, var)
        push!(unsafe_convert_exprs, quote
            $var = Base.unsafe_convert($(esc(typ)), $arg)
        end)
    end

    call = quote
        $(unsafe_convert_exprs...)

        gc_state = @ccall(jl_gc_safe_enter()::Int8)
        ret = ccall($(esc(func)), $(esc(rettype)), $(Expr(:tuple, map(esc, types)...)),
                    $(unsafe_convert_args...))
        @ccall(jl_gc_safe_leave(gc_state::Int8)::Cvoid)
        ret
    end

    quote
        $(cconvert_exprs...)

        GC.@preserve $(cconvert_args...)  $(call)
    end
end

"""
    @gcsafe_ccall ...

Call a foreign function just like `@ccall`, but marking it safe for the GC to run. This is
useful for functions that may block, so that the GC isn't blocked from running, but may also
be required to prevent deadlocks (see JuliaGPU/CUDA.jl#2261).

Note that this is generally only safe with non-Julia C functions that do not call back into
Julia. When using callbacks, the code should make sure to transition back into GC-unsafe
mode using the `@gcunsafe` macro.
"""
macro gcsafe_ccall(expr)
    ccall_macro_lower(Base.ccall_macro_parse(expr)...)
end

"""
    @gcunsafe_callback function callback(...)
        ...
    end

Mark a callback function as unsafe for the GC to run. This is normally the default for
Julia code, and is meant to be used in combination with `@gcsafe_ccall`.
"""
macro gcunsafe_callback(ex)
    if VERSION >= v"1.9"
        # on 1.9+, `@cfunction` already transitions to GC-unsafe mode
        return esc(ex)
    end

    # parse the function definition
    @assert Meta.isexpr(ex, :function)
    sig = ex.args[1]
    @assert Meta.isexpr(sig, :call)
    body = ex.args[2]
    @assert Meta.isexpr(body, :block)

    gcunsafe_body = quote
        gc_state = @ccall(jl_gc_unsafe_enter()::Int8)
        try
            $(ex)
        finally
            @ccall(jl_gc_unsafe_leave(gc_state::Int8)::Cvoid)
        end
    end

    return esc(Expr(:function, sig, gcunsafe_body))
end
