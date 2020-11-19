# ccall wrapper for calling functions in libraries that might not be available

export @checked, @workspace, @argout

"""
    @checked function foo(...)
        rv = ...
        return rv
    end

Macro for wrapping a function definition returning a status code. Two versions of the
function will be generated: `foo`, with the function body wrapped by an invocation of the
`@check` macro (to be implemented by the caller of this macro), and `unsafe_foo` where no
such invocation is present and the status code is returned to the caller.
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
        @check $body
    end
    safe_sig = Expr(:call, sig.args[1], sig.args[2:end]...)
    safe_def = Expr(:function, safe_sig, safe_body)

    # generate a "unsafe" version that returns the error code instead
    unsafe_sig = Expr(:call, Symbol("unsafe_", sig.args[1]), sig.args[2:end]...)
    unsafe_def = Expr(:function, unsafe_sig, body)

    return esc(:($safe_def, $unsafe_def))
end


"""
    @argout call(..., out(output_arg), ...)

Change the behavior of a function call, returning the value of an argument instead.
A common use case is to pass a newly-created Ref and immediately dereference that output:

    @argout(some_getter(Ref{Int}()))[]

If no output argument is specified, `nothing` will be returned.
Multiple output arguments return a tuple.

"""
macro argout(ex)
    Meta.isexpr(ex, :call) || throw(ArgumentError("@argout macro should be applied to a function call"))

    block = quote end

    # look for output arguments (`out(...)`)
    output_vars = []
    args = ex.args[2:end]
    for (i,arg) in enumerate(args)
        if Meta.isexpr(arg, :call) && arg.args[1] == :out
            # allocate a variable
            @gensym output_val
            push!(block.args, :($output_val = $(ex.args[i+1].args[2]))) # strip `output(...)`
            push!(output_vars, output_val)

            # replace the argument
            ex.args[i+1] = output_val
        end
    end

    # generate a return
    push!(block.args, ex)
    if isempty(output_vars)
        push!(block.args, :(nothing))
    elseif length(output_vars) == 1
        push!(block.args, :($(output_vars[1])))
    else
        push!(block.args, :(tuple($(output_vars...))))
    end

    esc(block)
end

"""
    @workspace size=getWorkspaceSize(args...) [eltyp=UInt8] [fallback=nothing] buffer->begin
      useWorkspace(workspace, sizeof(workspace))
    end

Create a GPU workspace vector with element type `eltyp` and size in number of elements (in
the default case of an UInt8 element type this equals to the amount of bytes) determined by
calling `getWorkspaceSize`, and pass it to the closure for use in calling `useWorkspace`.
A fallback workspace size `fallback` can be specified if the regular size would lead to OOM.
Afterwards, the buffer is put back into the memory pool for reuse.

This helper protects against the rare but real issue of `getWorkspaceSize` returning
different results based on the GPU device memory pressure, which might change _after_
initial allocation of the workspace (which can cause a GC collection).
"""
macro workspace(ex...)
    code = ex[end]
    kwargs = ex[1:end-1]

    sz = nothing
    eltyp = :UInt8
    fallback = nothing
    for kwarg in kwargs
        key,val = kwarg.args
        if key == :size
            sz = val
        elseif key == :eltyp
            eltyp = val
        elseif key == :fallback
            fallback = val
        else
            throw(ArgumentError("Unsupported keyword argument '$key'"))
        end
    end

    if sz === nothing
        throw(ArgumentError("@workspace macro needs a size argument"))
    end

    # break down the closure to a let block to prevent JuliaLang/julia#15276
    Meta.isexpr(code, :(->)) || throw(ArgumentError("@workspace macro should be applied to a closure"))
    length(code.args) == 2 || throw(ArgumentError("@workspace closure should take exactly one argument"))
    code_arg = code.args[1]
    code = code.args[2]

    return quote
        sz = $(esc(sz))
        workspace = nothing
        try
          while workspace === nothing || length(workspace) < sz
              workspace = CuArray{$(esc(eltyp))}(undef, sz)
              sz = $(esc(sz))
          end
        catch ex
            $fallback === nothing && rethrow()
            isa(ex, OutOfGPUMemoryError) || rethrow()
            workspace = CuArray{UInt8}(undef, $fallback)
        end

        let $(esc(code_arg)) = workspace
            ret = $(esc(code))
            CUDA.unsafe_free!(workspace)
            ret
        end
    end
end
