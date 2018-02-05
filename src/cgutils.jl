# Code generation utility functions

# how to map primitive Julia types to LLVM data types
const llvmtypes = IdDict{Type,Symbol}(
    Nothing => :void,
    Int8    => :i8,
    Int16   => :i16,
    Int32   => :i32,
    Int64   => :i64,
    UInt8   => :i8,
    UInt16  => :i16,
    UInt32  => :i32,
    UInt64  => :i64,
    Float32 => :float,
    Float64 => :double
)
const LLVMTypes = Union{keys(llvmtypes)...}     # for dispatch

# the inverse, ie. which Julia types map a given LLVM types
const jltypes = Dict{Symbol,Type}(
    :void   => Nothing,
    :i8     => Int8,
    :i16    => Int16,
    :i32    => Int32,
    :i64    => Int64,
    :float  => Float32,
    :double => Float64
)

# Decode an expression of the form:
#
#    function(arg::arg_type, arg::arg_type, ... arg::arg_type)::return_type
#
# Returns a tuple containing the function name, a vector of argument, a vector of argument
# types and the return type (all in symbolic form).
function decode_call(e)
    @assert e.head == :(::)

    # decode the return type expression: single symbol (the LLVM type), or a tuple of 2
    # symbols (the LLVM and corresponding Julia type)
    retspec = e.args[2]
    if isa(retspec, Symbol)
        rettype = retspec
    else
        @assert retspec.head == :tuple
        @assert length(retspec.args) == 2
        rettype = (retspec.args[1], retspec.args[2])
    end

    call = e.args[1]
    @assert call.head == :call

    fn = Symbol(call.args[1])
    args = Symbol[arg.args[1] for arg in call.args[2:end]]
    argtypes = Symbol[arg.args[2] for arg in call.args[2:end]]

    return fn, args, argtypes, rettype
end

# Generate a `llvmcall` statement calling an intrinsic specified as follows:
#
#     intrinsic(arg::arg_type, arg::arg_type, ... arg::arg_type)::return_type [attr]
#
# The argument types should be valid LLVM type identifiers (eg. i32, float, double).
# Conversions to the corresponding Julia type are automatically generated; make sure the
# actual arguments are of the same type to make these conversions no-ops. The optional
# argument `attr` indicates which LLVM function attributes (such as `readnone` or `nounwind`)
# to add to the intrinsic declaration.

# For example, the following call:
#     `@wrap __some_intrinsic(x::float, y::double)::float`
#
# will yield the following `llvmcall`:
# ```
#     Base.llvmcall(("declare float @__somme__intr(float, double)",
#                    "%3 = call float @__somme__intr(float %0, double %1)
#                     ret float %3"),
#                   Float32, Tuple{Float32,Float64},
#                   convert(Float32,x), convert(Float64,y))
# ```
macro wrap(call, attrs="")
    intrinsic, args, argtypes, rettype = decode_call(call)

    # decide on intrinsic return type
    if isa(rettype, Symbol)
        # only LLVM return type specified, match against known LLVM/Julia type combinations
        llvm_ret_typ = rettype
        julia_ret_typ = jltypes[rettype]
    else
        # both specified (for when there is a mismatch, eg. i32 -> UInt32)
        llvm_ret_typ = rettype[1]
        julia_ret_typ = rettype[2]
    end

    llvm_args = String["%$i" for i in 0:length(argtypes)]
    if llvm_ret_typ == :void
        llvm_ret_asgn = ""
        llvm_ret = "void"
    else
        llvm_ret_var = "%$(length(argtypes)+1)"
        llvm_ret_asgn = "$llvm_ret_var = "
        llvm_ret = "$llvm_ret_typ $llvm_ret_var"
    end
    llvm_declargs = join(argtypes, ", ")
    llvm_defargs = join(("$t $arg" for (t,arg) in zip(argtypes, llvm_args)), ", ")

    julia_argtypes = (jltypes[t] for t in argtypes)
    julia_args = (:(convert($argtype, $(esc(arg)))) for (arg, argtype) in zip(args, julia_argtypes))

    dest = ("""declare $llvm_ret_typ @$intrinsic($llvm_declargs)""",
            """$llvm_ret_asgn call $llvm_ret_typ @$intrinsic($llvm_defargs)
                ret $llvm_ret""")
    return quote
        Base.llvmcall($dest, $julia_ret_typ, Tuple{$(julia_argtypes...)}, $(julia_args...))
    end
end


# julia.h: jl_datatype_align
Base.@pure function datatype_align(::Type{T}) where {T}
    # typedef struct {
    #     uint32_t nfields;
    #     uint32_t alignment : 9;
    #     uint32_t haspadding : 1;
    #     uint32_t npointers : 20;
    #     uint32_t fielddesc_type : 2;
    # } jl_datatype_layout_t;
    field = T.layout + sizeof(UInt32)
    unsafe_load(convert(Ptr{UInt16}, field)) & convert(Int16, 2^9-1)
end


# generalization of word-based primitives

## extract a word from a value
@generated function extract_word(val, ::Val{i}) where {i}
    T_int32 = LLVM.Int32Type(jlctx[])

    bytes = Core.sizeof(val)
    T_val = convert(LLVMType, val)
    T_int = LLVM.IntType(8*bytes, jlctx[])

    # create function
    llvm_f, _ = create_function(T_int32, [T_val])
    mod = LLVM.parent(llvm_f)

    # generate IR
    Builder(jlctx[]) do builder
        entry = BasicBlock(llvm_f, "entry", jlctx[])
        position!(builder, entry)

        equiv = bitcast!(builder, parameters(llvm_f)[1], T_int)
        shifted = lshr!(builder, equiv, LLVM.ConstantInt(T_int, 32*(i-1)))
        # extracted = and!(builder, shifted, 2^32-1)
        extracted = trunc!(builder, shifted, T_int32, "word$i")

        ret!(builder, extracted)
    end

    call_function(llvm_f, UInt32, Tuple{val}, :( (val,) ))
end

## insert a word into a value
@generated function insert_word(val, word::UInt32, ::Val{i}) where {i}
    T_int32 = LLVM.Int32Type(jlctx[])

    bytes = Core.sizeof(val)
    T_val = convert(LLVMType, val)
    T_int = LLVM.IntType(8*bytes, jlctx[])

    # create function
    llvm_f, _ = create_function(T_val, [T_val, T_int32])
    mod = LLVM.parent(llvm_f)

    # generate IR
    Builder(jlctx[]) do builder
        entry = BasicBlock(llvm_f, "entry", jlctx[])
        position!(builder, entry)

        equiv = bitcast!(builder, parameters(llvm_f)[1], T_int)
        ext = zext!(builder, parameters(llvm_f)[2], T_int)
        shifted = shl!(builder, ext, LLVM.ConstantInt(T_int, 32*(i-1)))
        inserted = or!(builder, equiv, shifted)
        orig = bitcast!(builder, inserted, T_val)

        ret!(builder, orig)
    end

    call_function(llvm_f, val, Tuple{val, UInt32}, :( (val, word) ))
end

@generated function split_invocation(op::Function, val, args...)
    # TODO: control of lower-limit

    ex = quote
        Base.@_inline_meta
    end

    # disassemble into words
    words = Symbol[]
    for i in 1:Core.sizeof(val)÷4
        word = Symbol("word$i")
        push!(ex.args, :( $word = extract_word(val, Val($i)) ))
        push!(words, word)
    end

    # perform the operation
    for word in words
        push!(ex.args, :( $word = op($word, args...)) )
    end

    # reassemble
    push!(ex.args, :( out = zero(val) ))
    for (i,word) in enumerate(words)
        push!(ex.args, :( out = insert_word(out, $word, Val($i)) ))
    end

    push!(ex.args, :( out ))
    return ex
end

@generated function recurse_invocation(op::Function, val, args...)
    ex = quote
        Base.@_inline_meta
    end

    fields = fieldnames(val)
    if isempty(fields)
        push!(ex.args, :( split_invocation(op, val, args...) ))
    else
        ctor = Expr(:new, val)
        for field in fields
            push!(ctor.args, :( recurse_invocation(op, getfield(val, $(QuoteNode(field))),
                                                   args...) ))
        end
        push!(ex.args, ctor)
    end

    return ex
end