# Tools for implementing device functionality

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

## extract bits from a larger value
@inline function extract_word(val, ::Val{i}) where {i}
    extract_value(val, UInt32, Val(32*(i-1)))
end
@generated function extract_value(val, ::Type{sub}, ::Val{offset}) where {sub, offset}
    T_val = convert(LLVMType, val)
    T_sub = convert(LLVMType, sub)

    bytes = Core.sizeof(val)
    T_int = LLVM.IntType(8*bytes, JuliaContext())

    # create function
    llvm_f, _ = create_function(T_sub, [T_val])
    mod = LLVM.parent(llvm_f)

    # generate IR
    Builder(JuliaContext()) do builder
        entry = BasicBlock(llvm_f, "entry", JuliaContext())
        position!(builder, entry)

        equiv = bitcast!(builder, parameters(llvm_f)[1], T_int)
        shifted = lshr!(builder, equiv, LLVM.ConstantInt(T_int, offset))
        # extracted = and!(builder, shifted, 2^32-1)
        extracted = trunc!(builder, shifted, T_sub)

        ret!(builder, extracted)
    end

    call_function(llvm_f, UInt32, Tuple{val}, :( (val,) ))
end

## insert bits into a larger value
@inline function insert_word(val, word::UInt32, ::Val{i}) where {i}
    insert_value(val, word, Val(32*(i-1)))
end
@generated function insert_value(val, sub, ::Val{offset}) where {offset}
    T_val = convert(LLVMType, val)
    T_sub = convert(LLVMType, sub)

    bytes = Core.sizeof(val)
    T_out_int = LLVM.IntType(8*bytes, JuliaContext())

    # create function
    llvm_f, _ = create_function(T_val, [T_val, T_sub])
    mod = LLVM.parent(llvm_f)

    # generate IR
    Builder(JuliaContext()) do builder
        entry = BasicBlock(llvm_f, "entry", JuliaContext())
        position!(builder, entry)

        equiv = bitcast!(builder, parameters(llvm_f)[1], T_out_int)
        ext = zext!(builder, parameters(llvm_f)[2], T_out_int)
        shifted = shl!(builder, ext, LLVM.ConstantInt(T_out_int, offset))
        inserted = or!(builder, equiv, shifted)
        orig = bitcast!(builder, inserted, T_val)

        ret!(builder, orig)
    end

    call_function(llvm_f, val, Tuple{val, sub}, :( (val, sub) ))
end

# split the invocation of a function `op` on a value `val` with non-struct eltype
# into multiple smaller invocations on byte-sized partial values.
@generated function split_value_invocation(op::Function, val, args...)
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

# split the invocation of a function `op` on a value `val`
# by invoking the function on each of its fields
@generated function recurse_value_invocation(op::Function, val, args...)
    ex = quote
        Base.@_inline_meta
    end

    fields = fieldnames(val)
    if isempty(fields)
        push!(ex.args, :( split_value_invocation(op, val, args...) ))
    else
        ctor = Expr(:new, val)
        for field in fields
            push!(ctor.args, :(
                recurse_value_invocation(op, getfield(val, $(QuoteNode(field))), args...) ))
        end
        push!(ex.args, ctor)
    end

    return ex
end

# split the invocation of a function `op` on a pointer `ptr` with non-struct eltype
# into multiple smaller invocations on any supported pointer as listed in `supported_ptrs`.
@generated function split_pointer_invocation(op::Function, ptr, ::Type{supported_ptrs},
                                             args...) where {supported_ptrs}
    T = eltype(ptr)
    elsize(x) = Core.sizeof(eltype(x))
    supported_ptrs = reverse(Base.uniontypes(supported_ptrs))

    ex = quote
        Base.@_inline_meta
    end

    # disassemble
    vals = Tuple{Symbol,Int,Type}[]
    offset = 0
    while offset < Core.sizeof(T)
        val = Symbol("value.$(length(vals)+1)")

        # greedy selection of next pointer type
        remaining = Core.sizeof(T)-offset
        valid = filter(ptr->elsize(ptr)<=remaining, supported_ptrs)
        if isempty(valid)
            error("Cannot partition $T into values of $supported_typs")
        end
        ptr = first(sort(collect(valid); by=elsize, rev=true))

        push!(vals, (val, offset, ptr))
        offset += elsize(ptr)
    end

    # perform the operation
    for (val, offset, ptr) in vals
        subptr = :(convert($ptr, ptr+$offset))
        push!(ex.args, :( $val = op($subptr, args...)) )
    end

    # reassemble
    push!(ex.args, :( out = zero($T) ))
    for (val, offset, ptr) in vals
        push!(ex.args, :( out = insert_value(out, $val, Val($offset)) ))
    end

    push!(ex.args, :( out ))
    return ex
end

# split the invocation of a function `op` on a pointer `ptr`
# by invoking the function on a pointer to each of its fields
@generated function recurse_pointer_invocation(op::Function, ptr, ::Type{supported_ptrs},
                                               args...) where {supported_ptrs}
    T = eltype(ptr)

    ex = quote
        Base.@_inline_meta
    end

    fields = fieldnames(T)
    if isempty(fields)
        push!(ex.args, :( split_pointer_invocation(op, ptr, supported_ptrs, args...) ))
    else
        ctor = Expr(:new, T)
        for (i,field) in enumerate(fields)
            field_typ = fieldtype(T, i)
            field_offset = fieldoffset(T, i)
            field_ptr_typ = :($(ptr.name.wrapper){$field_typ})
            # NOTE: this ctor is a leap of faith
            subptr = :(convert($field_ptr_typ, ptr+$field_offset))
            push!(ctor.args, :(
                recurse_pointer_invocation(op, $subptr, supported_ptrs, args...) ))
        end
        push!(ex.args, ctor)
    end

    return ex
end
