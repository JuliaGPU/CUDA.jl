# Warp Shuffle (B.14)

# TODO: does not work on sub-word (ie. Int16) or non-word divisible sized types

# TODO: these functions should dispatch based on the actual warp size
const ws = Int32(32)

# TODO: this functionality should throw <sm_30


# primitive intrinsics

# "two packed values specifying a mask for logically splitting warps into sub-segments
# and an upper bound for clamping the source lane index"
@inline pack(width, mask) = (convert(UInt32, ws - width) << 8) | convert(UInt32, mask)

# NOTE: CUDA C disagrees with PTX on how shuffles are called
for (name, mode, mask, offset) in (("_up",   :up,   UInt32(0x00), src->src),
                                   ("_down", :down, UInt32(0x1f), src->src),
                                   ("_xor",  :bfly, UInt32(0x1f), src->src),
                                   ("",      :idx,  UInt32(0x1f), src->:($src-1)))
    fname = Symbol("shfl$name")
    @eval export $fname

    if cuda_driver_version >= v"9.0" && v"6.0" in ptx_support
        # newer hardware/CUDA versions use synchronizing intrinsics, which take an extra
        # mask argument indicating which threads in the lane should be synchronized
        intrinsic = "llvm.nvvm.shfl.sync.$mode.i32"

        fname_sync = Symbol("$(fname)_sync")
        __fname_sync = Symbol("__$(fname)_sync")
        @eval begin
            export $fname_sync

            # HACK: recurse_value_invocation and friends split the first argument of a call,
            #       so swap mask and val for these tools to works.
            @inline $fname_sync(mask, val, src, width=$ws) =
                $__fname_sync(val, mask, src, width)
            @inline $__fname_sync(val::UInt32, mask, src, width) =
                ccall($intrinsic, llvmcall, UInt32,
                      (UInt32, UInt32, UInt32, UInt32),
                      mask, val, $(offset(:src)), pack(width, $mask))

            # for backwards compatibility, have the non-synchronizing intrinsic dispatch
            # to the synchronizing one (with a full-lane default value for the mask)
            @inline $fname(val::UInt32, src, width=$ws, mask::UInt32=0xffffffff) =
                $fname_sync(mask, val, src, width)
        end
    else
        intrinsic = "llvm.nvvm.shfl.$mode.i32"

        @eval begin
            @inline $fname(val::UInt32, src, width=$ws) =
                ccall($intrinsic, llvmcall, UInt32,
                      (UInt32, UInt32, UInt32),
                      val, $(offset(:src)), pack(width, $mask))
        end
    end
end


# wide and aggregate intrinsics

for name in ["_up", "_down", "_xor", ""]
    fname = Symbol("shfl$name")
    @eval @inline $fname(src, args...) = recurse_value_invocation($fname, src, args...)

    fname_sync = Symbol("__$(fname)_sync")
    @eval @inline $fname_sync(src, args...) = recurse_value_invocation($fname_sync, src, args...)
end


# documentation

@doc """
    shfl(val, lane::Integer, width::Integer=32, threadmask::UInt32=0xffffffff)

Shuffle a value from a directly indexed lane `lane`. The argument `threadmask` for selecting
which threads to synchronize is only available on recent hardware, and defaults to all
threads in the warp.
""" shfl

@doc """
    shfl_up(val, delta::Integer, width::Integer=32, threadmask::UInt32=0xffffffff)

Shuffle a value from a lane with lower ID relative to caller. The argument `threadmask` for
selecting which threads to synchronize is only available on recent hardware, and defaults to
all threads in the warp.
""" shfl_up

@doc """
    shfl_down(val, delta::Integer, width::Integer=32, threadmask::UInt32=0xffffffff)

Shuffle a value from a lane with higher ID relative to caller. The argument `threadmask` for
selecting which threads to synchronize is only available on recent hardware, and defaults to
all threads in the warp.
""" shfl_down

@doc """
    shfl_xor(val, lanemask::Integer, width::Integer=32, threadmask::UInt32=0xffffffff)

Shuffle a value from a lane based on bitwise XOR of own lane ID with `lanemask`. The
argument `threadmask` for selecting which threads to synchronize is only available on recent
hardware, and defaults to all threads in the warp.
""" shfl_xor


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
