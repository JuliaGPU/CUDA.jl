# Warp Shuffle (B.14)

export FULL_MASK

# TODO: does not work on sub-word (ie. Int16) or non-word divisible sized types

# TODO: these functions should dispatch based on the actual warp size
const ws = Int32(32)

const FULL_MASK = 0xffffffff

# TODO: this functionality should throw <sm_30


# core intrinsics

# "two packed values specifying a mask for logically splitting warps into sub-segments
# and an upper bound for clamping the source lane index"
@inline pack(width, mask) = (convert(UInt32, ws - width) << 8) | convert(UInt32, mask)

# NOTE: CUDA C disagrees with PTX on how shuffles are called
for (name, mode, mask, offset) in (("_up",   :up,   UInt32(0x00), src->src),
                                   ("_down", :down, UInt32(0x1f), src->src),
                                   ("_xor",  :bfly, UInt32(0x1f), src->src),
                                   ("",      :idx,  UInt32(0x1f), src->:($src-1)))
    fname = Symbol("shfl$(name)_sync")
    @eval export $fname

    # LLVM intrinsics
    for (T,typ) in ((Int32, "i32"), (UInt32, "i32"), (Float32, "f32"))
        intrinsic = "llvm.nvvm.shfl.sync.$mode.$typ"
        @eval begin
            @inline $fname(mask, val::$T, src, width=$ws) =
                ccall($intrinsic, llvmcall, $T,
                    (UInt32, $T, UInt32, UInt32),
                    mask, val, $(offset(:src)), pack(width, $mask))
        end
    end
end


# extended versions

"""
    shfl_recurse(op, x::T)::T

Register how a shuffle operation `op` should be applied to a value `x` of type `T` that is
not natively supported by the shuffle intrinsics.
"""
shfl_recurse(op, x) = throw(ArgumentError("Unsupported value type for shuffle operation"))

for fname in (:shfl_up_sync, :shfl_down_sync, :shfl_xor_sync, :shfl_sync)
    @eval begin
        @inline $fname(mask, val, src, width=$ws) =
            shfl_recurse(x->$fname(mask, x, src, width), val)
    end
end

# unsigned integers
shfl_recurse(op, x::UInt8)   = op(UInt32(x)) % UInt8
shfl_recurse(op, x::UInt16)  = op(UInt32(x)) % UInt16
shfl_recurse(op, x::UInt64)  = (UInt64(op((x >>> 32) % UInt32)) << 32) | op((x & typemax(UInt32)) % UInt32)
shfl_recurse(op, x::UInt128) = (UInt128(op((x >>> 64) % UInt64)) << 64) | op((x & typemax(UInt64)) % UInt64)

# signed integers
shfl_recurse(op, x::Int8)  = reinterpret(Int8, shfl_recurse(op, reinterpret(UInt8, x)))
shfl_recurse(op, x::Int16)  = reinterpret(Int16, shfl_recurse(op, reinterpret(UInt16, x)))
shfl_recurse(op, x::Int64)  = reinterpret(Int64, shfl_recurse(op, reinterpret(UInt64, x)))
shfl_recurse(op, x::Int128) = reinterpret(Int128, shfl_recurse(op, reinterpret(UInt128, x)))

# floating point numbers
shfl_recurse(op, x::Float64) = reinterpret(Float64, shfl_recurse(op, reinterpret(UInt64, x)))

# other
shfl_recurse(op, x::Bool)    = op(UInt32(x)) % Bool
shfl_recurse(op, x::Complex) = Complex(op(real(x)), op(imag(x)))


# documentation

@doc """
    shfl_sync(threadmask::UInt32, val, lane::Integer, width::Integer=32)

Shuffle a value from a directly indexed lane `lane`, and synchronize threads according to
`threadmask`.
""" shfl_sync

@doc """
    shfl_up_sync(threadmask::UInt32, val, delta::Integer, width::Integer=32)

Shuffle a value from a lane with lower ID relative to caller, and synchronize threads
according to `threadmask`.
""" shfl_up_sync

@doc """
    shfl_down_sync(threadmask::UInt32, val, delta::Integer, width::Integer=32)

Shuffle a value from a lane with higher ID relative to caller, and synchronize threads
according to `threadmask`.
""" shfl_down_sync

@doc """
    shfl_xor_sync(threadmask::UInt32, val, mask::Integer, width::Integer=32)

Shuffle a value from a lane based on bitwise XOR of own lane ID with `mask`, and synchronize
threads according to `threadmask`.
""" shfl_xor_sync
