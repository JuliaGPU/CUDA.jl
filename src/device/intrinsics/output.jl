# Formatted Output (B.17)

export @cuprintf

const cuprintf_fmts = Vector{String}()

"""
Print a formatted string in device context on the host standard output:

    @cuprintf("%Fmt", args...)

Note that this is not a fully C-compliant `printf` implementation; see the CUDA
documentation for supported options and inputs.

Also beware that it is an untyped, and unforgiving `printf` implementation. Type widths need
to match, eg. printing a 64-bit Julia integer requires the `%ld` formatting string.
"""
macro cuprintf(fmt::String, args...)
    # NOTE: we can't pass fmt by Val{}, so save it in a global buffer
    push!(cuprintf_fmts, "$fmt\0")
    id = length(cuprintf_fmts)

    return :(generated_cuprintf(Val{$id}, $(map(esc, args)...)))
end

function emit_vprintf(id::Integer, argtypes, args...)
    fmt = cuprintf_fmts[id]
    fmtlen = length(fmt)

    llvm_argtypes = [llvmtypes[jltype] for jltype in argtypes]

    decls = Vector{String}()
    push!(decls, """declare i32 @vprintf(i8*, i8*)""")
    push!(decls, """%print$(id)_argtyp = type { $(join(llvm_argtypes, ", ")) }""")
    push!(decls, """@print$(id)_fmt = private unnamed_addr constant [$fmtlen x i8] c"$(escape_llvm_string(fmt))", align 1""")

    ir = Vector{String}()
    push!(ir, """%args = alloca %print$(id)_argtyp""")
    arg = 0
    tmp = length(args)+1
    for jltype in argtypes
        llvmtype = llvmtypes[jltype]
        push!(ir, """%$tmp = getelementptr inbounds %print$(id)_argtyp, %print$(id)_argtyp* %args, i32 0, i32 $arg""")
        push!(ir, """store $llvmtype %$arg, $llvmtype* %$tmp, align 4""")
        arg+=1
        tmp+=1
    end
    push!(ir, """%argptr = bitcast %print$(id)_argtyp* %args to i8*""")
    push!(ir, """%$tmp = call i32 @vprintf(i8* getelementptr inbounds ([$fmtlen x i8], [$fmtlen x i8]* @print$(id)_fmt, i32 0, i32 0), i8* %argptr)""")
    push!(ir, """ret void""")

    return quote
        Base.@_inline_meta
        Base.llvmcall(($(join(decls, "\n")),
                       $(join(ir,    "\n"))),
                      Void, Tuple{$argtypes...}, $(args...)
                     )
    end
end

@generated function generated_cuprintf{ID}(::Type{Val{ID}}, argspec...)
    args = [:( argspec[$i] ) for i in 1:length(argspec)]
    return emit_vprintf(ID, argspec, args...)
end
