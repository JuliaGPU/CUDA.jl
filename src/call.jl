# ccall wrapper for calling functions in libraries that might not be available

export @runtime_ccall

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
