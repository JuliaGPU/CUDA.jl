## random number generation

using Random
import RandomNumbers


# helpers

global_index() = (threadIdx().x, threadIdx().y, threadIdx().z,
                  blockIdx().x, blockIdx().y, blockIdx().z)


# global state

struct ThreadLocalXorshift32 <: RandomNumbers.AbstractRNG{UInt32}
    vals::CuDeviceArray{UInt32, 6, AS.Generic}
end

function init_random_state!(kernel, len)
    if kernel.random_state === missing || length(kernel.random_state) < len
        kernel.random_state = CuVector{UInt32}(undef, len)
    end

    random_state_ptr = CuGlobal{Ptr{Cvoid}}(kernel.mod, "global_random_state")
    random_state_ptr[] = reinterpret(Ptr{Cvoid}, pointer(kernel.random_state))
end

@eval @inline function global_random_state()
    ptr = reinterpret(LLVMPtr{UInt32, AS.Generic}, Base.llvmcall(
        $("""@global_random_state = weak externally_initialized global i$(WORD_SIZE) 0
             define i$(WORD_SIZE) @entry() #0 {
                 %ptr = load i$(WORD_SIZE), i$(WORD_SIZE)* @global_random_state, align 8
                 ret i$(WORD_SIZE) %ptr
             }
             attributes #0 = { alwaysinline }
          """, "entry"), Ptr{Cvoid}, Tuple{}))
    dims = (blockDim().x, blockDim().y, blockDim().z, gridDim().x, gridDim().y, gridDim().z)
    CuDeviceArray(dims, ptr)
end

@device_override Random.default_rng() = ThreadLocalXorshift32(global_random_state())

@device_override Random.make_seed() = clock(UInt32)

function Random.seed!(rng::ThreadLocalXorshift32, seed::Integer)
    index = global_index()
    rng.vals[index...] = seed % UInt32
    return
end


# generators

function xorshift(x::UInt32)::UInt32
    x = xor(x, x << 13)
    x = xor(x, x >> 17)
    x = xor(x, x << 5)
    return x
end

function Random.rand(rng::ThreadLocalXorshift32, ::Type{UInt32})
    # NOTE: we add the current linear index to the local state, to make sure threads get
    #       different random numbers when unseeded (initial state = 0 for all threads)
    index = global_index()
    offset = LinearIndices(rng.vals)[index...]
    state = rng.vals[index...] + UInt32(offset)

    new_state = xorshift(state)
    rng.vals[index...] = new_state
    return new_state
end
