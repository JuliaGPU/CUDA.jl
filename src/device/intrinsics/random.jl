## random number generation

using Random


# helpers

global_index() = (threadIdx().x, threadIdx().y, threadIdx().z,
                  blockIdx().x, blockIdx().y, blockIdx().z)


# global state

struct ThreadLocalRNG <: AbstractRNG
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

@device_override Random.default_rng() = ThreadLocalRNG(global_random_state())

@device_override Random.make_seed() = clock(UInt32)

function Random.seed!(rng::ThreadLocalRNG, seed::Integer)
    index = global_index()
    rng.vals[index...] = seed
    return
end


# generators

function xorshift(x::UInt32)::UInt32
    x = xor(x, x << 13)
    x = xor(x, x >> 17)
    x = xor(x, x << 5)
    return x
end

function get_thread_word(rng::ThreadLocalRNG)
    # NOTE: we add the current linear index to the local state, to make sure threads get
    #       different random numbers when unseeded (initial state = 0 for all threads)
    index = global_index()
    offset = LinearIndices(rng.vals)[index...]
    state = rng.vals[index...] + UInt32(offset)

    new_state = generate_next_state(state)
    rng.vals[index...] = new_state

    return new_state    # FIXME: return old state?
end

function generate_next_state(state::UInt32)
    new_val = xorshift(state)
    return UInt32(new_val)
end

# TODO: support for more types (can we reuse more of the Random standard library?)
#       see RandomNumbers.jl

function Random.rand(rng::ThreadLocalRNG, ::Type{Float32})
    word = get_thread_word(rng)
    res = (word >> 9) | reinterpret(UInt32, 1f0)
    return reinterpret(Float32, res) - 1.0f0
end
