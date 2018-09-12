# Warp Shuffle (B.14)

# TODO: does not work on sub-word (ie. Int16) or non-word divisible sized types

# TODO: should shfl_idx conform to 1-based indexing?

# TODO: these functions should dispatch based on the actual warp size
const ws = Int32(32)

# TODO: this functionality should throw <sm_30


# primitive intrinsics

# "two packed values specifying a mask for logically splitting warps into sub-segments
# and an upper bound for clamping the source lane index"
@inline pack(width::UInt32, mask::UInt32)::UInt32 = (convert(UInt32, ws - width) << 8) | mask

# NOTE: CUDA C disagrees with PTX on how shuffles are called
for (name, mode, mask) in (("_up",   :up,   UInt32(0x00)),
                           ("_down", :down, UInt32(0x1f)),
                           ("_xor",  :bfly, UInt32(0x1f)),
                           ("",      :idx,  UInt32(0x1f)))
    fname = Symbol("shfl$name")

    if cuda_driver_version >= v"9.0" && v"6.0" in ptx_support
        instruction = Symbol("shfl.sync.$mode.b32")
        fname_sync = Symbol("$(fname)_sync")

        # TODO: implement using LLVM intrinsics when we have D38090

        @eval begin
            export $fname_sync, $fname

            @inline $fname_sync(val::UInt32, src::UInt32, width::UInt32=$ws,
                                threadmask::UInt32=0xffffffff) =
                @asmcall($"$instruction \$0, \$1, \$2, \$3, \$4;", "=r,r,r,r,r", true,
                         UInt32, NTuple{4,UInt32},
                         val, src, pack(width, $mask), threadmask)

            # FIXME: replace this with a checked conversion once we have exceptions
            @inline $fname_sync(val::UInt32, src::Integer, width::Integer=$ws,
                                threadmask::UInt32=0xffffffff) =
                $fname_sync(val, unsafe_trunc(UInt32, src), unsafe_trunc(UInt32, width),
                            threadmask)

            @inline $fname(val::UInt32, src::Integer, width::Integer=$ws) =
                $fname_sync(val, src, width)
        end
    else
        intrinsic = Symbol("llvm.nvvm.shfl.$mode.i32")

        @eval begin
            export $fname
            @inline $fname(val::UInt32, src::UInt32, width::UInt32=$ws) =
                ccall($"$intrinsic", llvmcall, UInt32,
                      (UInt32, UInt32, UInt32),
                      val, src, pack(width, $mask))

            # FIXME: replace this with a checked conversion once we have exceptions
            @inline $fname(val::UInt32, src::Integer, width::Integer=$ws) =
                $fname(val, unsafe_trunc(UInt32, src), unsafe_trunc(UInt32, width))
        end
    end
end


# wide and aggregate intrinsics

for name in ["_up", "_down", "_xor", ""]
    fname = Symbol("shfl$name")
    @eval @inline $fname(src, args...) = recurse_value_invocation($fname, src, args...)

    fname_sync = Symbol("$(fname)_sync")
    @eval @inline $fname_sync(src, args...) = recurse_value_invocation($fname, src, args...)
end


# documentation

@doc """
    shfl(val, lane::Integer, width::Integer=32)

Shuffle a value from a directly indexed lane `lane`.
""" shfl

@doc """
    shfl_up(val, delta::Integer, width::Integer=32)

Shuffle a value from a lane with lower ID relative to caller.
""" shfl_up

@doc """
    shfl_down(val, delta::Integer, width::Integer=32)

Shuffle a value from a lane with higher ID relative to caller.
""" shfl_down

@doc """
    shfl_xor(val, mask::Integer, width::Integer=32)

Shuffle a value from a lane based on bitwise XOR of own lane ID with `mask`.
""" shfl_xor


@doc """
    shfl_sync(val, lane::Integer, width::Integer=32, threadmask::UInt32=0xffffffff)

Shuffle a value from a directly indexed lane `lane`. The default value for `threadmask`
performs the shuffle on all threads in the warp.
""" shfl_sync

@doc """
    shfl_up_sync(val, delta::Integer, width::Integer=32, threadmask::UInt32=0xffffffff)

Shuffle a value from a lane with lower ID relative to caller. The default value for
`threadmask` performs the shuffle on all threads in the warp.
""" shfl_up_sync

@doc """
    shfl_down_sync(val, delta::Integer, width::Integer=32, threadmask::UInt32=0xffffffff)

Shuffle a value from a lane with higher ID relative to caller. The default value for
`threadmask` performs the shuffle on all threads in the warp.
""" shfl_down_sync

@doc """
    shfl_xor_sync(val, mask::Integer, width::Integer=32, threadmask::UInt32=0xffffffff)

Shuffle a value from a lane based on bitwise XOR of own lane ID with `mask`. The default
value for `threadmask` performs the shuffle on all threads in the warp.
""" shfl_xor_sync
