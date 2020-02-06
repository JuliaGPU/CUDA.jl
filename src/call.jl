# ccall wrapper for calling functions in libraries that might not be available

export @runtime_ccall, decode_ccall_function, @checked

"""
    @runtime_ccall((function_name, library), returntype, (argtype1, ...), argvalue1, ...)

Extension of `ccall` that performs the lookup of `function_name` in `library` at run time.
This is useful in the case that `library` might not be available, in which case a function
that performs a `ccall` to that library would fail to compile.

After a slower first call to load the library and look up the function, no additional
overhead is expected compared to regular `ccall`.
"""
macro runtime_ccall(target, args...)
    # decode ccall function/library target
    Meta.isexpr(target, :tuple) || error("Expected (function_name, library) tuple")
    function_name, library = target.args

    # global const ref to hold the function pointer
    @gensym fptr_cache
    @eval __module__ begin
        const $fptr_cache = Ref(C_NULL)
    end

    return quote
        # use a closure to hold the lookup and avoid code bloat in the caller
        @noinline function lookup_fptr()
            library = Libdl.dlopen($(esc(library)))
            $(esc(fptr_cache))[] = Libdl.dlsym(library, $(esc(function_name)))

            $(esc(fptr_cache))[]
        end

        fptr = $(esc(fptr_cache))[]
        if fptr == C_NULL   # folded into the null check performed by ccall
            fptr = lookup_fptr()
        end

        ccall(fptr, $(map(esc, args)...))
    end

    return
end

# decode `ccall` or `@runtime_ccall` invocations and extract the function that is called
function decode_ccall_function(ex)
    # check is used in front of `ccall` or `@runtime_ccall`s that work on a tuple (fun, lib)
    if Meta.isexpr(ex, :call)
        @assert ex.args[1] == :ccall
        @assert Meta.isexpr(ex.args[2], :tuple)
        fun = String(ex.args[2].args[1].value)
    elseif Meta.isexpr(ex, :macrocall)
        @assert ex.args[1] == Symbol("@runtime_ccall")
        @assert Meta.isexpr(ex.args[3], :tuple)
        fun = String(ex.args[3].args[1].value)
    else
        error("@check should prefix ccall or @runtime_ccall")
    end

    # strip any version tag (e.g. cuEventDestroy_v2 -> cuEventDestroy)
    m = match(r"_v\d+$", fun)
    if m !== nothing
        fun = fun[1:end-length(m.match)]
    end

    return fun
end

"""
    @checked function foo(...)
        ccall(...)
    end

Macro for wrapping a function definition containing a single `ccall` expression. Two
versions of the function will be generated: `foo`, with the `ccall` wrapped by an invocation
of the `@check` macro (to be implemented by the caller of this macro), and `unsafe_foo`
where no such invocation is present.
"""
macro checked(ex)
    # parse the function definition
    @assert Meta.isexpr(ex, :function)
    sig = ex.args[1]
    @assert Meta.isexpr(sig, :call)
    body = ex.args[2]
    @assert Meta.isexpr(body, :block)
    @assert length(body.args) == 2      # line number node and a single call

    # generate a "safe" version that performs a check
    safe_body = Expr(:block, body.args[1], :(@check $(body.args[2])))
    safe_sig = Expr(:call, sig.args[1], sig.args[2:end]...)
    safe_def = Expr(:function, safe_sig, safe_body)

    # generate a "unsafe" version that returns the error code instead
    unsafe_sig = Expr(:call, Symbol("unsafe_", sig.args[1]), sig.args[2:end]...)
    unsafe_def = Expr(:function, unsafe_sig, body)

    return esc(:($safe_def, $unsafe_def))
end
