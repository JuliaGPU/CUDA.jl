module PoolUtils

using ..CUDA

using TimerOutputs

using Printf


export MEMDEBUG

const MEMDEBUG = ccall(:jl_is_memdebug, Bool, ())


## locking

export NonReentrantLock, @spinlock, @lock

const var"@lock" = Base.var"@lock"

# a simple non-reentrant lock that errors when trying to reenter on the same task
struct NonReentrantLock <: Threads.AbstractLock
  rl::ReentrantLock
  NonReentrantLock() = new(ReentrantLock())
end

function Base.lock(nrl::NonReentrantLock)
  @assert !islocked(nrl.rl) || nrl.rl.locked_by !== current_task()
  lock(nrl.rl)
end

function Base.trylock(nrl::NonReentrantLock)
  @assert !islocked(nrl.rl) || nrl.rl.locked_by !== current_task()
  trylock(nrl.rl)
end

Base.unlock(nrl::NonReentrantLock) = unlock(nrl.rl)

# a safe way to acquire locks from finalizers, where we can't wait (which switches tasks)
macro spinlock(l, ex)
  quote
    temp = $(esc(l))
    while !trylock(temp)
      ccall(:jl_cpu_pause, Cvoid, ())
      # Temporary solution before we have gc transition support in codegen.
      ccall(:jl_gc_safepoint, Cvoid, ())
      # we can't yield here
    end
    try
      $(esc(ex))
    finally
      unlock(temp)
    end
  end
end


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


## timing

export @pool_timeit

const to = TimerOutput()

macro pool_timeit(args...)
    TimerOutputs.timer_expr(CUDA, true, :($CUDA.to), args...)
end

end
