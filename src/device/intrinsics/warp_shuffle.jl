# Warp Shuffle (B.14)

# TODO: should shfl_idx conform to 1-based indexing?

# narrow
for typ in ((Int32,   :i32, :i32),
            (UInt32,  :i32, :i32),
            (Float32, :f32, :float))
    jl, intr, llvm = typ

    # TODO: these functions should dispatch based on the actual warp size
    ws = Int32(32)

    # NOTE: CUDA C disagrees with PTX on how shuffles are called
    for (fname, mode, mask) in ((:shfl_up,   :up,   UInt32(0x00)),
                                (:shfl_down, :down, UInt32(0x1f)),
                                (:shfl_xor,  :bfly, UInt32(0x1f)),
                                (:shfl,      :idx,  UInt32(0x1f)))
        pack_expr = :((($ws - convert(UInt32, width)) << 8) | $mask)
        intrinsic = Symbol("llvm.nvvm.shfl.$mode.$intr")

        @eval begin
            export $fname
            @inline $fname(val::$jl, srclane::Integer, width::Integer=$ws) =
                ccall($"$intrinsic", llvmcall, $jl,
                      ($jl, UInt32, UInt32), val, convert(UInt32, srclane), $pack_expr)
        end

    end
end

@inline decode(val::UInt64) = trunc(UInt32,  val & 0x00000000ffffffff),
                              trunc(UInt32, (val & 0xffffffff00000000)>>32)

@inline encode(x::UInt32, y::UInt32) = UInt64(x) | UInt64(y)<<32

# wide
# NOTE: we only reuse the i32 shuffle, does it make any difference using eg. f32 shuffle for f64 values?
for typ in (Int64, UInt64, Float64)
    # TODO: these functions should dispatch based on the actual warp size
    ws = Int32(32)

    for mode in (:up, :down, :bfly, :idx)
        fname = Symbol("shfl_$mode")
        @eval begin
            export $fname
            @inline function $fname(val::$typ, srclane::Integer, width::Integer=$ws)
                x,y = decode(reinterpret(UInt64, val))
                x = $fname(x, srclane, width)
                y = $fname(y, srclane, width)
                reinterpret($typ, encode(x,y))
            end
        end
    end
end

@doc """
    shfl_idx(val, src::Integer, width::Integer=32)

Shuffle a value from a directly indexed lane `src`
""" shfl

@doc """
    shfl_up(val, src::Integer, width::Integer=32)

Shuffle a value from a lane with lower ID relative to caller.
""" shfl_up

@doc """
    shfl_down(val, src::Integer, width::Integer=32)

Shuffle a value from a lane with higher ID relative to caller.
""" shfl_down

@doc """
    shfl_xor(val, src::Integer, width::Integer=32)

Shuffle a value from a lane based on bitwise XOR of own lane ID.
""" shfl_xor
