# ccall wrapper for calling functions in libraries that might not be available

export @checked, with_workspace, @debug_ccall

"""
    @checked function foo(...)
        rv = ...
        return rv
    end

Macro for wrapping a function definition returning a status code. Two versions of the
function will be generated: `foo`, with the function execution wrapped by an invocation of
the `check` function (to be implemented by the caller of this macro), and `unsafe_foo` where
no such invocation is present and the status code is returned to the caller.
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

    # generate a "unsafe" version that returns the error code instead
    unsafe_sig = Expr(:call, Symbol("unsafe_", sig.args[1]), sig.args[2:end]...)
    unsafe_def = Expr(:function, unsafe_sig, body)

    return esc(:($safe_def, $unsafe_def))
end

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
@inline with_workspace(f, size::Union{Integer,Function}, fallback=nothing; kwargs...) =
    with_workspace(f, UInt8, isa(size, Integer) ? ()->size : size, fallback; kwargs...)

function with_workspace(f::Function, eltyp::Type{T}, size::Union{Integer,Function},
                        fallback::Union{Nothing,Integer}=nothing; keep::Bool=false) where {T}
    get_size() = Int(isa(size, Integer) ? size : size()::Integer)

    # allocate
    sz = get_size()
    workspace = nothing
    try
        while workspace === nothing || length(workspace) < sz
            workspace = CuVector{T}(undef, sz)
            sz = get_size()
        end
    catch ex
        fallback === nothing && rethrow()
        isa(ex, OutOfGPUMemoryError) || rethrow()
        workspace = CuVector{T}(undef, fallback)
    end
    workspace = workspace::CuVector{T}

    # use & free
    try
        f(workspace)
    finally
        keep || CUDA.unsafe_free!(workspace)
    end
end

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
