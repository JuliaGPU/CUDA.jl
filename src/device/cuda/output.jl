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
            argtypes = LLVM.StructType("printf_args", JuliaContext())
            elements!(argtypes, param_types)

            args = alloca!(builder, argtypes)
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


## print-like functionality

export @cuprint, @cuprintln

# simple conversions, defining an expression and the resulting argument type. nothing fancy,
# `@cuprint` pretty directly maps to `@cuprintf`; we should just support `write(::IO)`.
const cuprint_conversions = Dict(
    Float32     => (x->:(Float64($x)),             Float64),
    Ptr{<:Any}  => (x->:(convert(Ptr{Cvoid}, $x)), Ptr{Cvoid})
)

# format specifiers
const cuprint_specifiers = Dict(
    # integers
    Int16       => "%hd",
    Int32       => "%d",
    Int64       => Sys.iswindows() ? "%lld" : "%ld",
    UInt16      => "%hu",
    UInt32      => "%u",
    UInt64      => Sys.iswindows() ? "%llu" : "%lu",

    # floating-point
    Float64     => "%f",

    # other
    Cchar       => "%c",
    Ptr{Cvoid}  => "%p",
)

@generated function _cuprint(parts...)
    fmt = ""
    args = Expr[]

    for i in 1:length(parts)
        part = :(parts[$i])
        T = parts[i]

        # put literals directly in the format string
        if T <: Val
            fmt *= string(T.parameters[1])
            continue
        end

        # try to convert arguments if they are not supported directly
        if !haskey(cuprint_specifiers, T)
            for Tmatch in keys(cuprint_conversions)
                if T <: Tmatch
                    conv, T = cuprint_conversions[Tmatch]
                    part = conv(part)
                    break
                end
            end
        end

        # render the argument
        if haskey(cuprint_specifiers, T)
            fmt *= cuprint_specifiers[T]
            push!(args, part)
        elseif T <: String
            @error("@cuprint does not support non-literal strings")
        else
            @error("@cuprint does not support values of type $T")
        end
    end

    quote
        Base.@_inline_meta
        @cuprintf($fmt, $(args...))
    end
end

"""
    @cuprint(xs...)
    @cuprintln(xs...)

Print a textual representation of values `xs` to standard output from the GPU. The
functionality builds on `@cuprintf`, and is intended as a more use friendly alternative of
that API. However, that also means there's only limited support for argument types, handling
16/32/64 signed and unsigned integers, 32 and 64-bit floating point numbers, chars and
pointers. For more complex output, use `@cuprintf` directly.

Limited string interpolation is also possible:

    @cuprint("Hello, World ", 42, "\n")
    @cuprint "Hello, World $(42)\n"

"""
macro cuprint(parts...)
    args = Union{Val,Expr,Symbol}[]

    parts = [parts...]
    while true
        isempty(parts) && break

        part = popfirst!(parts)

        # handle string interpolation
        if isa(part, Expr) && part.head == :string
            parts = vcat(part.args, parts)
            continue
        end

        # expose literals to the generator by using Val types
        if isbits(part) # literal numbers, etc
            push!(args, Val(part))
        elseif isa(part, QuoteNode) # literal symbols
            push!(args, Val(part.value))
        elseif isa(part, String) # literal strings need to be interned
            push!(args, Val(Symbol(part)))
        else # actual values that will be passed to printf
            push!(args, part)
        end
    end

    quote
        _cuprint($(map(esc, args)...))
    end
end

@doc (@doc @cuprint) ->
macro cuprintln(parts...)
    parts = map(part -> isa(part, Expr) || isa(part, Symbol) ? esc(part) : part, parts)
    quote
        @cuprint($(parts...), "\n")
    end
end
