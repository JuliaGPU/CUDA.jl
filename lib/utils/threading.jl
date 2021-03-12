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
