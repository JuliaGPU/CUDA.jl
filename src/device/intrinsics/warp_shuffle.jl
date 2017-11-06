# Warp Shuffle (B.14)

# TODO: does not work on sub-word (ie. Int16) or non-word divisible sized types

# TODO: should shfl_idx conform to 1-based indexing?

# TODO: these functions should dispatch based on the actual warp size
const ws = Int32(32)


# single-word primitives

# NOTE: CUDA C disagrees with PTX on how shuffles are called
for (name, mode, mask) in (("_up",   :up,   UInt32(0x00)),
                           ("_down", :down, UInt32(0x1f)),
                           ("_xor",  :bfly, UInt32(0x1f)),
                           ("",      :idx,  UInt32(0x1f)))
    fname = Symbol("shfl$name")

    # "two packed values specifying a mask for logically splitting warps into sub-segments
    # and an upper bound for clamping the source lane index"
    pack_expr = :(((convert(UInt32, $ws - width)) << 8) | $mask)

    if cuda_version >= v"9.0-" && VERSION >= v"0.7.0-DEV.1959"
        instruction = Symbol("shfl.sync.$mode.b32")
        fname_sync = Symbol("$(fname)_sync")

        # TODO: implement using LLVM intrinsics when we have D38090

        @eval begin
            export $fname_sync, $fname

            @inline $fname_sync(val::UInt32, src::Integer, width::Integer=$ws,
                                threadmask::UInt32=0xffffffff) =
                Base.llvmcall(
                    $"""%5 = call i32 asm sideeffect "$instruction \$0, \$1, \$2, \$3, \$4;", "=r,r,r,r,r"(i32 %0, i32 %1, i32 %2, i32 %3)
                        ret i32 %5""",
                    UInt32, NTuple{4,UInt32}, val, src, $pack_expr, threadmask)

            @inline $fname(val::UInt32, src::Integer, width::Integer=$ws) =
                $fname_sync(val, src, width)
        end
    else
        intrinsic = Symbol("llvm.nvvm.shfl.$mode.i32")

        @eval begin
            export $fname
            @inline $fname(val::UInt32, src::Integer, width::Integer=$ws) =
                ccall($"$intrinsic", llvmcall, UInt32,
                      (UInt32, UInt32, UInt32), val, convert(UInt32, src), $pack_expr)
        end
    end
end


# multi-word primitives (recurse into words)

## extract a word from a value
@generated function extract_word(val, ::Val{i}) where {i}
    T_int32 = LLVM.Int32Type(jlctx[])

    bytes = Core.sizeof(val)
    T_val = convert(LLVMType, val)
    T_int = LLVM.IntType(8*bytes, jlctx[])

    # create function
    llvmf = create_llvmf(T_int32, [T_val])
    mod = LLVM.parent(llvmf)

    # generate IR
    Builder(jlctx[]) do builder
        entry = BasicBlock(llvmf, "entry", jlctx[])
        position!(builder, entry)

        equiv = bitcast!(builder, parameters(llvmf)[1], T_int)
        shifted = lshr!(builder, equiv, LLVM.ConstantInt(T_int, 32*(i-1)))
        # extracted = and!(builder, shifted, 2^32-1)
        extracted = trunc!(builder, shifted, T_int32, "word$i")

        ret!(builder, extracted)
    end

    call_llvmf(llvmf, UInt32, Tuple{val}, :( (val,) ))
end

## insert a word into a value
@generated function insert_word(val, word::UInt32, ::Val{i}) where {i}
    T_int32 = LLVM.Int32Type(jlctx[])

    bytes = Core.sizeof(val)
    T_val = convert(LLVMType, val)
    T_int = LLVM.IntType(8*bytes, jlctx[])

    # create function
    llvmf = create_llvmf(T_val, [T_val, T_int32])
    mod = LLVM.parent(llvmf)

    # generate IR
    Builder(jlctx[]) do builder
        entry = BasicBlock(llvmf, "entry", jlctx[])
        position!(builder, entry)

        equiv = bitcast!(builder, parameters(llvmf)[1], T_int)
        ext = zext!(builder, parameters(llvmf)[2], T_int)
        shifted = shl!(builder, ext, LLVM.ConstantInt(T_int, 32*(i-1)))
        inserted = or!(builder, equiv, shifted)
        orig = bitcast!(builder, inserted, T_val)

        ret!(builder, orig)
    end

    call_llvmf(llvmf, val, Tuple{val, UInt32}, :( (val, word) ))
end

@generated function shuffle_primitive(op::Function, val, args...)
    ex = quote
        Base.@_inline_meta
    end

    # disassemble into words
    words = Symbol[]
    for i in 1:Core.sizeof(val)÷4
        word = Symbol("word$i")
        push!(ex.args, :( $word = extract_word(val, Val{$i}()) ))
        push!(words, word)
    end

    # shuffle
    for word in words
        push!(ex.args, :( $word = op($word, args...)) )
    end

    # reassemble
    push!(ex.args, :( out = zero(val) ))
    for (i,word) in enumerate(words)
        push!(ex.args, :( out = insert_word(out, $word, Val{$i}()) ))
    end

    push!(ex.args, :( out ))
    return ex
end


# aggregates (recurse into fields)

@generated function shuffle_aggregate(op::Function, val::T, args...) where T
    ex = quote
        Base.@_inline_meta
    end

    fields = fieldnames(T)
    if isempty(fields)
        push!(ex.args, :( shuffle_primitive(op, val, args...) ))
    else
        ctor = Expr(:new, T)
        for field in fields
            push!(ctor.args, :( shuffle_aggregate(op, getfield(val, $(QuoteNode(field))),
                                                  args...) ))
        end
        push!(ex.args, ctor)
    end

    return ex
end


# entry-point functions

for name in ["_up", "_down", "_xor", ""]
    fname = Symbol("shfl$name")
    @eval @inline $fname(src, args...) = shuffle_aggregate($fname, src, args...)

    fname_sync = Symbol("$(fname)_sync")
    @eval @inline $fname_sync(src, args...) = shuffle_aggregate($fname, src, args...)
end


# documentation

@doc """
    shfl(val, lane::Integer, width::Integer=32)
    shfl_sync(val, lane::Integer, width::Integer=32, threadmask::UInt32=0xffffffff)

Shuffle a value from a directly indexed lane `lane`.
""" shfl
@doc (@doc shfl) shfl_sync

@doc """
    shfl_up(val, delta::Integer, width::Integer=32)
    shfl_up_sync(val, delta::Integer, width::Integer=32, threadmask::UInt32=0xffffffff)

Shuffle a value from a lane with lower ID relative to caller.
""" shfl_up
@doc (@doc shfl_up) shfl_up_sync

@doc """
    shfl_down(val, delta::Integer, width::Integer=32)
    shfl_down_sync(val, delta::Integer, width::Integer=32, threadmask::UInt32=0xffffffff)

Shuffle a value from a lane with higher ID relative to caller.
""" shfl_down
@doc (@doc shfl_down) shfl_down_sync

@doc """
    shfl_xor(val, mask::Integer, width::Integer=32)
    shfl_xor_sync(val, mask::Integer, width::Integer=32, threadmask::UInt32=0xffffffff)

Shuffle a value from a lane based on bitwise XOR of own lane ID with `mask`.
""" shfl_xor
@doc (@doc shfl_xor) shfl_xor_sync
