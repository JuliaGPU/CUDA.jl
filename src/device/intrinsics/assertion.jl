# Assertion (B.19)

export @cuassert

"""
    @assert cond [text]

Signal assertion failure to the CUDA driver if `cond` is `false`. Preferred syntax for
writing assertions, mimicking `Base.@assert`. Message `text` is optionally displayed upon
assertion failure.

!!! warning
    A failed assertion will crash the GPU, so use sparingly as a debugging tool.
    Furthermore, the assertion might be disabled at various optimization levels, and thus
    should not cause any side-effects.
"""
macro cuassert(ex, msgs...)
    # message handling copied from Base.@assert
    msg = isempty(msgs) ? ex : msgs[1]
    if isa(msg, AbstractString)
        msg = msg # pass-through
    elseif !isempty(msgs) && (isa(msg, Expr) || isa(msg, Symbol))
        # message is an expression needing evaluating
        msg = :(Main.Base.string($(esc(msg))))
    elseif isdefined(Main, :Base) && isdefined(Main.Base, :string) && applicable(Main.Base.string, msg)
        msg = Main.Base.string(msg)
    else
        # string() might not be defined during bootstrap
        msg = :(Main.Base.string($(Expr(:quote,msg))))
    end

    return :($(esc(ex)) ? $(nothing)
                        : cuassert_fail($(Val(Symbol(msg))),
                                        $(Val(__source__.file)),
                                        $(Val(__source__.line))))
end

assert_counter = 0

@generated function cuassert_fail(::Val{msg}, ::Val{file}, ::Val{line}) where
                                 {msg, file, line}
    T_void = LLVM.VoidType(JuliaContext())
    T_int32 = LLVM.Int32Type(JuliaContext())
    T_pint8 = LLVM.PointerType(LLVM.Int8Type(JuliaContext()))

    # create function
    llvm_f, _ = create_function()
    mod = LLVM.parent(llvm_f)

    # generate IR
    Builder(JuliaContext()) do builder
        entry = BasicBlock(llvm_f, "entry", JuliaContext())
        position!(builder, entry)

        global assert_counter
        assert_counter += 1

        message = globalstring_ptr!(builder, String(msg), "assert_message_$(assert_counter)")
        file = globalstring_ptr!(builder, String(file), "assert_file_$(assert_counter)")
        line = ConstantInt(T_int32, line)
        func = globalstring_ptr!(builder, "unknown", "assert_function_$(assert_counter)")
        charSize = ConstantInt(Csize_t(1), JuliaContext())

        # invoke __assertfail and return
        # NOTE: we don't mark noreturn since that control flow might confuse ptxas
        assertfail_typ =
            LLVM.FunctionType(T_void,
                             [T_pint8, T_pint8, T_int32, T_pint8, llvmtype(charSize)])
        assertfail = LLVM.Function(mod, "__assertfail", assertfail_typ)
        call!(builder, assertfail, [message, file, line, func, charSize])

        ret!(builder)
    end

    call_function(llvm_f, Nothing, Tuple{})
end
