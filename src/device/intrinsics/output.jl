# Formatted Output (B.17)

export @cuprintf

@generated function promote_c_argument(arg)
    # > When a function with a variable-length argument list is called, the variable
    # > arguments are passed using C's old ``default argument promotions.'' These say that
    # > types char and short int are automatically promoted to int, and type float is
    # > automatically promoted to double. Therefore, varargs functions will never receive
    # > arguments of type char, short int, or float.

    if arg == Cchar || arg == Cshort
        return :(Cint(arg))
    elseif arg == Cfloat
        return :(Cdouble(arg))
    else
        return :(arg)
    end
end

"""
Print a formatted string in device context on the host standard output:

    @cuprintf("%Fmt", args...)

Note that this is not a fully C-compliant `printf` implementation; see the CUDA
documentation for supported options and inputs.

Also beware that it is an untyped, and unforgiving `printf` implementation. Type widths need
to match, eg. printing a 64-bit Julia integer requires the `%ld` formatting string.
"""
macro cuprintf(fmt::String, args...)
    fmt_val = Val(Symbol(fmt))

    return :(_cuprintf($fmt_val, $(map(arg -> :(promote_c_argument($arg)), esc.(args))...)))
end

@generated function _cuprintf(::Val{fmt}, argspec...) where {fmt}
    arg_exprs = [:( argspec[$i] ) for i in 1:length(argspec)]
    arg_types = [argspec...]

    T_void = LLVM.VoidType(JuliaContext())
    T_int32 = LLVM.Int32Type(JuliaContext())
    T_pint8 = LLVM.PointerType(LLVM.Int8Type(JuliaContext()))

    # create functions
    param_types = LLVMType[convert.(LLVMType, arg_types)...]
    llvm_f, _ = create_function(T_int32, param_types)
    mod = LLVM.parent(llvm_f)

    # generate IR
    Builder(JuliaContext()) do builder
        entry = BasicBlock(llvm_f, "entry", JuliaContext())
        position!(builder, entry)

        str = globalstring_ptr!(builder, String(fmt))

        # construct and fill args buffer
        if isempty(argspec)
            buffer = LLVM.PointerNull(T_pint8)
        else
            argtypes = LLVM.StructType("vprintf_args", JuliaContext())
            elements!(argtypes, param_types)

            args = alloca!(builder, argtypes, "args")
            for (i, param) in enumerate(parameters(llvm_f))
                p = struct_gep!(builder, args, i-1)
                store!(builder, param, p)
            end

            buffer = bitcast!(builder, args, T_pint8)
        end

        # invoke vprintf and return
        vprintf_typ = LLVM.FunctionType(T_int32, [T_pint8, T_pint8])
        vprintf = LLVM.Function(mod, "vprintf", vprintf_typ)
        chars = call!(builder, vprintf, [str, buffer])

        ret!(builder, chars)
    end

    arg_tuple = Expr(:tuple, arg_exprs...)
    call_function(llvm_f, Int32, Tuple{arg_types...}, arg_tuple)
end
