## random number generation

using Random
import RandomNumbers


# global state

# we provide 4KB of random state, shared across the thread block using shared memory.
# 128 bytes of seeding data is available in constant memory, set during initial load.

@eval @inline function global_random_seed()
    ptr = Base.llvmcall(
        $("""@global_random_seed = weak addrspace($(AS.Constant)) externally_initialized global [32 x i32] zeroinitializer, align 32
             define i8 addrspace($(AS.Constant))* @entry() #0 {
                 %ptr = getelementptr inbounds [32 x i32], [32 x i32] addrspace($(AS.Constant))* @global_random_seed, i64 0, i64 0
                 %untyped_ptr = bitcast i32 addrspace($(AS.Constant))* %ptr to i8 addrspace($(AS.Constant))*
                 ret i8 addrspace($(AS.Constant))* %untyped_ptr
             }
             attributes #0 = { alwaysinline }
          """, "entry"), LLVMPtr{UInt32, AS.Constant}, Tuple{})
    CuDeviceArray((32,), ptr)
end

@eval @inline function global_random_state()
    ptr = Base.llvmcall(
        $("""@global_random_state = weak addrspace($(AS.Shared)) global [1024 x i32] zeroinitializer, align 32
             define i8 addrspace($(AS.Shared))* @entry() #0 {
                 %ptr = getelementptr inbounds [1024 x i32], [1024 x i32] addrspace($(AS.Shared))* @global_random_state, i64 0, i64 0
                 %untyped_ptr = bitcast i32 addrspace($(AS.Shared))* %ptr to i8 addrspace($(AS.Shared))*
                 ret i8 addrspace($(AS.Shared))* %untyped_ptr
             }
             attributes #0 = { alwaysinline }
          """, "entry"), LLVMPtr{UInt32, AS.Shared}, Tuple{})
    CuDeviceArray((1024,), ptr)
end

function initialize_random_seeds!(mod)
    seed = CuGlobal{NTuple{32,UInt32}}(mod, "global_random_seed")
    seed[async=true] = Tuple(rand(UInt32, 32))
end

@device_override Random.make_seed() = clock(UInt32)

@generated function make_full_seed(seed::Integer)
    quote
        s_0 = seed % UInt32
        Base.@nexprs 32 i->s_i = xorshift(s_{i-1})
        Base.@ncall 32 tuple s
    end
end


# helpers

function xorshift(x::UInt32)::UInt32
    x = xor(x, x << 13)
    x = xor(x, x >> 17)
    x = xor(x, x << 5)
    return x
end


# generators

"""
    SharedTauswortheGenerator()

A maximally equidistributed combined Tausworthe generator.

The generator uses 32 bytes of random state per warp, 1024 bytes total, stored in shared
memory. This memory is zero-initialized; When the first random number is generated, each
warp will derive an initial state from the seed provided during module compilation, and the
block and warp identifiers. Each warp will then indepentendly, but deterministically use
that state to generate random numbers. Finally, the first thread of each warp updates the
shared state.

!!! warning

    Although the numbers obtained from this generator "look OK", they do not pass the
    SmallCrush test suite, so the generator should be deemed experimental.
"""
struct SharedTauswortheGenerator <: RandomNumbers.AbstractRNG{UInt32}
end

@inline function Base.getproperty(rng::SharedTauswortheGenerator, field::Symbol)
    if field === :seed
        global_random_seed()
    elseif field === :state
        # return a warp-local view of the global random state
        threadId = UInt32(threadIdx().x + (threadIdx().y - 1) * blockDim().x +
                                        (threadIdx().z - 1) * blockDim().x * blockDim().y)
        warpId = (threadId-UInt32(1)) >> 5 + UInt32(1)  # fld1
        warpOffset = 32*(warpId-1)+1
        view(global_random_state(), warpOffset:warpOffset+32)
    end
end

@device_override Random.default_rng() = SharedTauswortheGenerator()

function Random.seed!(rng::SharedTauswortheGenerator, seed::Integer)
    Random.seed!(rng, make_full_seed(seed))
    return
end

"""
    Random.seed!(rng::SharedTauswortheGenerator, seeds)

Seed the on-device Tausworthe generator with a 32-element tuple or array of UInt32 seeds.

!!! warning

    This function should be called by all threads to ensure all warp-local state is set.
"""
@inline Base.@propagate_inbounds function Random.seed!(rng::SharedTauswortheGenerator, seed)
    state = initial_state(seed)
    @inbounds rng.state[laneid()] = state
    return
end

@inline Base.@propagate_inbounds function initial_state(seeds)
    z = seeds[laneid()]

    # mix-in the warp and block id to ensure unique values across blocks
    # XXX: is this OK? shouldn't we use a generator that allows skipping ahead?
    #      https://stackoverflow.com/questions/11692785/efficient-xorshift-skip-ahead
    blockId = blockIdx().x + (blockIdx().y - 1) * gridDim().x +
                             (blockIdx().z - 1) * gridDim().x * gridDim().y
    threadId = UInt32(threadIdx().x + (threadIdx().y - 1) * blockDim().x +
                                      (threadIdx().z - 1) * blockDim().x * blockDim().y)
    warpId = (threadId-UInt32(1)) >> 5 + UInt32(1)  # fld1
    z = xorshift(z ⊻ blockId%UInt32 ⊻ (warpId << 16))

    return z
end

# NOTE: @propagate_inbounds for when we know the passed seed contains at least 32 elements
# XXX: without @inline @propagate_inbounds, we get local memory accesses (e.g. with `rand!`)

#const TausShift1 = CuConstantMemory{UInt32}((6, 2, 13, 3))
#const TausShift2 = CuConstantMemory{UInt32}((13, 27, 21, 12))
#const TausShift3 = CuConstantMemory{UInt32}((18, 2, 7, 13))
#const TausOffset = CuConstantMemory{UInt32}((4294967294, 4294967288, 4294967280, 4294967168))
# XXX: constant memory isn't supported yet, so let's resort to llvmcall
for (name, vals) in ("TausShift1" => (6, 2, 13, 3),
                     "TausShift2" => (13, 27, 21, 12),
                     "TausShift3" => (18, 2, 7, 13),
                     "TausOffset" => (4294967294, 4294967288, 4294967280, 4294967168))
    @eval @inline function $(Symbol(name))()
        ptr = Base.llvmcall(
            $("""@$(name) = weak addrspace($(AS.Constant)) global [4 x i32] [i32 $(vals[1]), i32 $(vals[2]), i32 $(vals[3]), i32 $(vals[4])], align 4
                define i8 addrspace($(AS.Constant))* @entry() #0 {
                    %ptr = getelementptr inbounds [4 x i32], [4 x i32] addrspace($(AS.Constant))* @$(name), i64 0, i64 0
                    %untyped_ptr = bitcast i32 addrspace($(AS.Constant))* %ptr to i8 addrspace($(AS.Constant))*
                    ret i8 addrspace($(AS.Constant))* %untyped_ptr
                }
                attributes #0 = { alwaysinline }
            """, "entry"), LLVMPtr{UInt32, AS.Constant}, Tuple{})
        CuDeviceArray((4,), ptr)
    end
end

function TausStep(z::Unsigned, S1::Integer, S2::Integer, S3::Integer, M::Unsigned)
    b = (((z << S1) ⊻ z) >> S2)
    return (((z & M) << S3) ⊻ b)
end

"""
    Random.rand(rng::SharedTauswortheGenerator, UInt32)

Generate a byte of random data using the on-device Tausworthe generator.
"""
function Random.rand(rng::SharedTauswortheGenerator, ::Type{UInt32})
    @inline pow2_mod1(x, y) = (x-1)&(y-1) + 1
    i = pow2_mod1(laneid(), 4)

    @inbounds begin
        # get state
        z = rng.state[laneid()]
        if z == 0
            z = initial_state(rng.seed)
        end

        # advance & update state
        S1, S2, S3, M = TausShift1()[i], TausShift2()[i], TausShift3()[i], TausOffset()[i]
        state = TausStep(z, S1, S2, S3, M)
        rng.state[laneid()] = state

        # generate based on 4 bytes of state
        i1 = pow2_mod1(laneid()+1, 32)
        i2 = pow2_mod1(laneid()+2, 32)
        i3 = pow2_mod1(laneid()+3, 32)
        if active_mask() == typemax(UInt32)
            # we have a full warp, so we can safely shuffle.
            # get the warp-local states from neighbouring threads.
            s1 = shfl_sync(FULL_MASK, state, i1)
            s2 = shfl_sync(FULL_MASK, state, i2)
            s3 = shfl_sync(FULL_MASK, state, i3)
        else
            # can't shuffle, so fall back to fetching global state.
            s1 = rng.state[i1]
            s2 = rng.state[i2]
            s3 = rng.state[i3]
        end
        state ⊻ s1 ⊻ s2 ⊻ s3
    end
end
