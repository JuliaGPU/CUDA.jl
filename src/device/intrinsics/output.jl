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

function cuprintf!(builder::Builder, fmt::String, params...)
    block = position(builder)
    llvm_f = LLVM.parent(block)
    mod = LLVM.parent(llvm_f)
    ctx = context(mod)

    T_int32 = LLVM.Int32Type(ctx)
    T_pint8 = LLVM.PointerType(LLVM.Int8Type(ctx))

    str = globalstring_ptr!(builder, fmt)

    # construct and fill args buffer
    if isempty(params)
        buffer = LLVM.PointerNull(T_pint8)
    else
        param_types = LLVMType[llvmtype.(params)...]
        argtypes = LLVM.StructType("printf_args", ctx)
        elements!(argtypes, param_types)

        args = alloca!(builder, argtypes)
        for (i, param) in enumerate(params)
            p = struct_gep!(builder, args, i-1)
            store!(builder, param, p)
        end

        buffer = bitcast!(builder, args, T_pint8)
    end

    vprintf = if haskey(functions(mod), "vprintf")
        functions(mod)["vprintf"]
    else
        vprintf_typ = LLVM.FunctionType(T_int32, [T_pint8, T_pint8])
        LLVM.Function(mod, "vprintf", vprintf_typ)
    end

    call!(builder, vprintf, [str, buffer])
end

@generated function _cuprintf(::Val{fmt}, argspec...) where {fmt}
    arg_exprs = [:( argspec[$i] ) for i in 1:length(argspec)]
    arg_types = [argspec...]

    T_int32 = LLVM.Int32Type(JuliaContext())

    # create the function
    param_types = LLVMType[convert.(LLVMType, arg_types)...]
    llvm_f, _ = create_function(T_int32, param_types)

    # generate IR
    Builder(JuliaContext()) do builder
        entry = BasicBlock(llvm_f, "entry", JuliaContext())
        position!(builder, entry)

        chars = cuprintf!(builder, String(fmt), parameters(llvm_f)...)

        ret!(builder, chars)
    end

    # invoke the function
    arg_tuple = Expr(:tuple, arg_exprs...)
    call_function(llvm_f, Int32, Tuple{arg_types...}, arg_tuple)
end
