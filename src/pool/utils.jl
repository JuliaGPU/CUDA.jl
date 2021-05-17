module PoolUtils

using ..CUDA

using Printf


export MEMDEBUG

const MEMDEBUG = ccall(:jl_is_memdebug, Bool, ())


## block of memory

export Block, INVALID, AVAILABLE, ALLOCATED, FREED, iswhole

@enum BlockState begin
    INVALID
    AVAILABLE
    ALLOCATED
    FREED
end

mutable struct Block
    buf::Mem.DeviceBuffer # base allocation
    sz::Int               # size into it
    off::Int              # offset into it

    state::BlockState
    prev::Union{Nothing,Block}
    next::Union{Nothing,Block}

    Block(buf, sz; off=0, state=INVALID, prev=nothing, next=nothing) =
        new(buf, sz, off, state, prev, next)
end

Base.sizeof(block::Block) = block.sz
Base.pointer(block::Block) = pointer(block.buf) + block.off

iswhole(block::Block) = block.prev === nothing && block.next === nothing

function Base.show(io::IO, block::Block)
    fields = [@sprintf("%p", Int(pointer(block)))]
    push!(fields, Base.format_bytes(sizeof(block)))
    push!(fields, "$(block.state)")
    block.off != 0 && push!(fields, "offset=$(block.off)")
    block.prev !== nothing && push!(fields, "prev=Block(offset=$(block.prev.off))")
    block.next !== nothing && push!(fields, "next=Block(offset=$(block.next.off))")

    print(io, "Block(", join(fields, ", "), ")")
end

end
