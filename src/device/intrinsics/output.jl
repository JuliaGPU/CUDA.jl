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
    push!(cuprintf_fmts, "$fmt")
    id = length(cuprintf_fmts)

    return :(_cuprintf(Val{$id}, $(map(esc, args)...)))
end

@generated function _cuprintf(::Type{Val{id}}, argspec...) where {id}
    arg_exprs = [:( argspec[$i] ) for i in 1:length(argspec)]
    arg_types = [argspec...]

    # TODO: needs to adhere to C vararg promotion (short -> int, float -> double, etc.)

    T_void = LLVM.VoidType(jlctx[])
    T_int32 = LLVM.Int32Type(jlctx[])
    T_pint8 = LLVM.PointerType(LLVM.Int8Type(jlctx[]))

    # create functions
    param_types = LLVMType[convert.(LLVMType, arg_types)...]
    llvmf = create_llvmf(T_int32, param_types)
    mod = LLVM.parent(llvmf)

    # generate IR
    Builder(jlctx[]) do builder
        entry = BasicBlock(llvmf, "entry", jlctx[])
        position!(builder, entry)

        fmt = globalstring_ptr!(builder, cuprintf_fmts[id])

        # construct and fill args buffer
        if isempty(argspec)
            buffer = LLVM.PointerNull(T_pint8)
        else
            argtypes = LLVM.StructType("vprintf_args", jlctx[])
            elements!(argtypes, param_types)

            args = alloca!(builder, argtypes, "args")
            for (i, param) in enumerate(parameters(llvmf))
                p = struct_gep!(builder, args, i-1)
                store!(builder, param, p)
            end

            buffer = bitcast!(builder, args, T_pint8)
        end

        # invoke vprintf and return
        vprintf_typ = LLVM.FunctionType(T_int32, [T_pint8, T_pint8])
        vprintf = LLVM.Function(mod, "vprintf", vprintf_typ)
        chars = call!(builder, vprintf, [fmt, buffer])

        ret!(builder, chars)
    end

    arg_tuple = Expr(:tuple, arg_exprs...)
    call_llvmf(llvmf, Int32, Tuple{arg_types...}, arg_tuple)
end
