## random number generation

using Random
import RandomNumbers


# global state

# shared memory with the actual seed, per warp, loaded lazily or overridden by calling `seed!`
@eval @inline function global_random_keys()
    ptr = Base.llvmcall(
        $("""@global_random_keys = weak addrspace($(AS.Shared)) global [32 x i32] zeroinitializer, align 32
             define i8 addrspace($(AS.Shared))* @entry() #0 {
                 %ptr = getelementptr inbounds [32 x i32], [32 x i32] addrspace($(AS.Shared))* @global_random_keys, i64 0, i64 0
                 %untyped_ptr = bitcast i32 addrspace($(AS.Shared))* %ptr to i8 addrspace($(AS.Shared))*
                 ret i8 addrspace($(AS.Shared))* %untyped_ptr
             }
             attributes #0 = { alwaysinline }
          """, "entry"), LLVMPtr{UInt32, AS.Shared}, Tuple{})
    CuDeviceArray((32,), ptr)
end

# shared memory with per-warp counters, incremented when generating numbers
@eval @inline function global_random_counters()
    ptr = Base.llvmcall(
        $("""@global_random_counters = weak addrspace($(AS.Shared)) global [32 x i32] zeroinitializer, align 32
             define i8 addrspace($(AS.Shared))* @entry() #0 {
                 %ptr = getelementptr inbounds [32 x i32], [32 x i32] addrspace($(AS.Shared))* @global_random_counters, i64 0, i64 0
                 %untyped_ptr = bitcast i32 addrspace($(AS.Shared))* %ptr to i8 addrspace($(AS.Shared))*
                 ret i8 addrspace($(AS.Shared))* %untyped_ptr
             }
             attributes #0 = { alwaysinline }
          """, "entry"), LLVMPtr{UInt32, AS.Shared}, Tuple{})
    CuDeviceArray((32,), ptr)
end

@device_override Random.make_seed() = clock(UInt32)


# generators

using Random123: philox2x_round, philox2x_bumpkey

# GPU-compatible/optimized version of the generator from Random123.jl
struct Philox2x32{R} <: RandomNumbers.AbstractRNG{UInt64}
    @inline function Philox2x32{R}() where R
        rng = new{R}()
        if rng.key == 0
            # initialize the key. this happens when first accessing the (0-initialized)
            # shared memory key from each block. if we ever want to make the device seed
            # controlable from the host, this would be the place to read a global seed.
            #
            # note however that it is undefined how shared memory persists across e.g.
            # launches, so we may not be able to rely on the zero initalization then.
            rng.key = Random.make_seed()
        end
        return rng
    end
end

# default to 7 rounds; enough to pass SmallCrush
@inline Philox2x32() = Philox2x32{7}()

@inline function Base.getproperty(rng::Philox2x32, field::Symbol)
    threadId = threadIdx().x + (threadIdx().y - 1i32) * blockDim().x +
                               (threadIdx().z - 1i32) * blockDim().x * blockDim().y
    warpId = (threadId - 1i32) >> 0x5 + 1i32  # fld1

    if field === :seed
        @inbounds global_random_seed()[1]
    elseif field === :key
        @inbounds global_random_keys()[warpId]
    elseif field === :ctr1
        @inbounds global_random_counters()[warpId]
    elseif field === :ctr2
        blockId = blockIdx().x + (blockIdx().y - 1i32) * gridDim().x +
                                 (blockIdx().z - 1i32) * gridDim().x * gridDim().y
        globalId = threadId + (blockId - 1i32) * (blockDim().x * blockDim().y * blockDim().z)
        globalId%UInt32
    end::UInt32
end

@inline function Base.setproperty!(rng::Philox2x32, field::Symbol, x)
    threadId = threadIdx().x + (threadIdx().y - 1i32) * blockDim().x +
                               (threadIdx().z - 1i32) * blockDim().x * blockDim().y
    warpId = (threadId - 1i32) >> 0x5 + 1i32  # fld1

    if field === :key
        @inbounds global_random_keys()[warpId] = x
    elseif field === :ctr1
        @inbounds global_random_counters()[warpId] = x
    end
end

@device_override @inline Random.default_rng() = Philox2x32()

"""
    Random.seed!(rng::Philox2x32, seed::Integer, [counter::Integer=0])

Seed the on-device Philox2x32 generator with an UInt32 number.
Should be called by at least one thread per warp.
"""
function Random.seed!(rng::Philox2x32, seed::Integer, counter::Integer=0)
    rng.key = seed % UInt32
    rng.ctr1 = counter
    return
end

if VERSION >= v"1.7-"
@device_override Random.seed!(::Random._GLOBAL_RNG, seed) =
    Random.seed!(Random.default_rng(), seed)
end

"""
    Random.rand(rng::Philox2x32, UInt32)

Generate a byte of random data using the on-device Tausworthe generator.
"""
function Random.rand(rng::Philox2x32{R},::Type{UInt64}) where {R}
    ctr1, ctr2, key = rng.ctr1, rng.ctr2, rng.key

    if R > 0                               ctr1, ctr2 = philox2x_round(ctr1, ctr2, key); end
    if R > 1  key = philox2x_bumpkey(key); ctr1, ctr2 = philox2x_round(ctr1, ctr2, key); end
    if R > 2  key = philox2x_bumpkey(key); ctr1, ctr2 = philox2x_round(ctr1, ctr2, key); end
    if R > 3  key = philox2x_bumpkey(key); ctr1, ctr2 = philox2x_round(ctr1, ctr2, key); end
    if R > 4  key = philox2x_bumpkey(key); ctr1, ctr2 = philox2x_round(ctr1, ctr2, key); end
    if R > 5  key = philox2x_bumpkey(key); ctr1, ctr2 = philox2x_round(ctr1, ctr2, key); end
    if R > 6  key = philox2x_bumpkey(key); ctr1, ctr2 = philox2x_round(ctr1, ctr2, key); end
    if R > 7  key = philox2x_bumpkey(key); ctr1, ctr2 = philox2x_round(ctr1, ctr2, key); end
    if R > 8  key = philox2x_bumpkey(key); ctr1, ctr2 = philox2x_round(ctr1, ctr2, key); end
    if R > 9  key = philox2x_bumpkey(key); ctr1, ctr2 = philox2x_round(ctr1, ctr2, key); end
    if R > 10 key = philox2x_bumpkey(key); ctr1, ctr2 = philox2x_round(ctr1, ctr2, key); end
    if R > 11 key = philox2x_bumpkey(key); ctr1, ctr2 = philox2x_round(ctr1, ctr2, key); end
    if R > 12 key = philox2x_bumpkey(key); ctr1, ctr2 = philox2x_round(ctr1, ctr2, key); end
    if R > 13 key = philox2x_bumpkey(key); ctr1, ctr2 = philox2x_round(ctr1, ctr2, key); end
    if R > 14 key = philox2x_bumpkey(key); ctr1, ctr2 = philox2x_round(ctr1, ctr2, key); end
    if R > 15 key = philox2x_bumpkey(key); ctr1, ctr2 = philox2x_round(ctr1, ctr2, key); end

    # update the warp counter
    # NOTE: this performs the same update on every thread in the warp, but each warp writes
    #       to a unique location so the duplicate writes are innocuous
    # XXX: what if this overflows? we can't increment ctr2. bump the key?
    rng.ctr1 += 1i32

    # NOTE: it's too expensive to keep both numbers around in case the user only wanted one,
    #       so just make our 2x32 generator return 64-bit numbers by default.
    return (ctr1 % UInt64) << 32 | (ctr2 % UInt64)
end
