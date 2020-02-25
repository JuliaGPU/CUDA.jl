# Tools for implementing device functionality


function tbaa_make_child(name::String, constant::Bool=false; ctx::LLVM.Context=JuliaContext())
    tbaa_root = MDNode([MDString("ptxtbaa", ctx)], ctx)
    tbaa_struct_type =
        MDNode([MDString("ptxtbaa_$name", ctx),
                tbaa_root,
                LLVM.ConstantInt(0, ctx)], ctx)
    tbaa_access_tag =
        MDNode([tbaa_struct_type,
                tbaa_struct_type,
                LLVM.ConstantInt(0, ctx),
                LLVM.ConstantInt(constant ? 1 : 0, ctx)], ctx)

    return tbaa_access_tag
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
