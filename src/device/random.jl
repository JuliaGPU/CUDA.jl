## random number generation

using Random
import RandomNumbers


# global state

# we cannot store RNG state in thread-local memory (i.e. in the `rng` object) because that
# inflate register usage. instead, we store it in shared memory, with one entry per warp.
#
# XXX: this implies that state is shared between `rng` objects, which can be surprising.

# array with seeds, per warp, initialized on kernel start or by calling `seed!`
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
    CuDeviceArray{UInt32,1,AS.Shared,Int32}(ptr, (32,))
end

# array with per-warp counters, incremented when generating numbers
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
    CuDeviceArray{UInt32,1,AS.Shared,Int32}(ptr, (32,))
end

# initialization function, called automatically at the start of each kernel because
# there's no reliable way to detect uninitialized shared memory (see JuliaGPU/CUDA.jl#2008)
function initialize_rng_state()
    threadId = threadIdx().x + (threadIdx().y - 1i32) * blockDim().x +
                               (threadIdx().z - 1i32) * blockDim().x * blockDim().y
    warpId = (threadId - 1i32) >> 0x5 + 1i32  # fld1

    @inbounds global_random_keys()[warpId] = kernel_state().random_seed
    @inbounds global_random_counters()[warpId] = 0
end

# generators

using Random123: philox2x_round, philox2x_bumpkey

# GPU-compatible/optimized version of the generator from Random123.jl
struct Philox2x32{R} <: RandomNumbers.AbstractRNG{UInt64}
    # NOTE: the state is stored globally; see comments at the top of this file.
end

# default to 7 rounds; enough to pass BigCrush
@inline Philox2x32() = Philox2x32{7}()

@inline function Base.getproperty(rng::Philox2x32, field::Symbol)
    threadId = threadIdx().x + (threadIdx().y - 1i32) * blockDim().x +
                               (threadIdx().z - 1i32) * blockDim().x * blockDim().y
    warpId = (threadId - 1i32) >> 0x5 + 1i32  # fld1

    if field === :key
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

if VERSION >= v"1.11-"
    # `Random.seed!(::AbstractRNG)` now passes a `nothing` seed value
    Random.seed!(rng::Philox2x32, seed::Nothing) =
        Random.seed!(rng, clock(UInt32))
else
    # ... where it used to call `Random_make_seed()`
    @device_override Random.make_seed() = clock(UInt32)
end

# seeding the implicit default RNG
if VERSION >= v"1.11-"
    @device_override Random.seed!(seed) =
        Random.seed!(Random.default_rng(), seed)
else
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
    # NOTE: this is not guaranteed to be visible in other kernels (JuliaGPU/CUDA.jl#2008)
    # XXX: what if this overflows? we can't increment ctr2. bump the key?
    rng.ctr1 += 1i32

    # NOTE: it's too expensive to keep both numbers around in case the user only wanted one,
    #       so just make our 2x32 generator return 64-bit numbers by default.
    return (ctr1 % UInt64) << 32 | (ctr2 % UInt64)
end



# normally distributed random numbers using Ziggurat algorithm
#
# copied from Base because we don't support its global tables

# a hacky method of exposing constant tables as constant GPU memory
function emit_constant_array(name::Symbol, data::AbstractArray{T}) where {T}
    @dispose ctx=Context() begin
        T_val = convert(LLVMType, T)
        T_ptr = convert(LLVMType, LLVMPtr{T,AS.Constant})

        # define function and get LLVM module
        llvm_f, _ = create_function(T_ptr)
        mod = LLVM.parent(llvm_f)

        # create a global memory global variable
        # TODO: global_var alignment?
        T_global = LLVM.ArrayType(T_val, length(data))
        # XXX: why can't we use a single name like emit_shmem
        gv = GlobalVariable(mod, T_global, "gpu_$(name)_data", AS.Constant)
        alignment!(gv, 16)
        linkage!(gv, LLVM.API.LLVMInternalLinkage)
        initializer!(gv, ConstantArray(data))

        # generate IR
        @dispose builder=IRBuilder() begin
            entry = BasicBlock(llvm_f, "entry")
            position!(builder, entry)

            ptr = gep!(builder, T_global, gv, [ConstantInt(0), ConstantInt(0)])

            untyped_ptr = bitcast!(builder, ptr, T_ptr)

            ret!(builder, untyped_ptr)
        end

        call_function(llvm_f, LLVMPtr{T,AS.Constant})
    end
end

for var in [:ki, :wi, :fi, :ke, :we, :fe]
    val = getfield(Random, var)
    gpu_var = Symbol("gpu_$var")
    arr_typ = :(CuDeviceArray{$(eltype(val)),$(ndims(val)),AS.Constant,Int32})
    @eval @inline @generated function $gpu_var()
        ptr = emit_constant_array($(QuoteNode(var)), $val)
        Expr(:call, $arr_typ, ptr, $(size(val)))
    end
end

## randn

@device_override Random.randn(rng::AbstractRNG) =
    _randn(rng, Random.rand(rng, Random.UInt52Raw()))

@inline function _randn(rng::AbstractRNG, r::UInt64)
    @inbounds begin
        r &= 0x000fffffffffffff
        rabs = Int64(r>>1) # One bit for the sign
        idx = rabs & 0xFF
        x = ifelse(r % Bool, -rabs, rabs)*gpu_wi()[idx+1]
        rabs < gpu_ki()[idx+1] && return x # 99.3% of the time we return here 1st try
        return randn_unlikely(rng, idx, rabs, x)
    end
end

# this unlikely branch is put in a separate function for better efficiency
@noinline function randn_unlikely(rng, idx, rabs, x)
    @inbounds if idx == 0
        while true
            xx = -Random.ziggurat_nor_inv_r*log(Random.rand(rng))
            yy = -log(Random.rand(rng))
            yy+yy > xx*xx &&
                return (rabs >> 8) % Bool ? -Random.ziggurat_nor_r-xx : Random.ziggurat_nor_r+xx
        end
    elseif (gpu_fi()[idx] - gpu_fi()[idx+1])*Random.rand(rng) + gpu_fi()[idx+1] < exp(-0.5*x*x)
        return x # return from the triangular area
    else
        return Random.randn(rng)
    end
end

## randexp

@device_override Random.randexp(rng::AbstractRNG) =
    _randexp(rng, Random.rand(rng, Random.UInt52Raw()))

function _randexp(rng::AbstractRNG, ri::UInt64)
    @inbounds begin
        ri &= 0x000fffffffffffff
        idx = ri & 0xFF
        x = ri*gpu_we()[idx+1]
        ri < gpu_ke()[idx+1] && return x # 98.9% of the time we return here 1st try
        return randexp_unlikely(rng, idx, x)
    end
end

@noinline function randexp_unlikely(rng, idx, x)
    @inbounds if idx == 0
        return Random.ziggurat_exp_r - log(Random.rand(rng))
    elseif (gpu_fe()[idx] - gpu_fe()[idx+1])*Random.rand(rng) + gpu_fe()[idx+1] < exp(-x)
        return x # return from the triangular area
    else
        return Random.randexp(rng)
    end
end
