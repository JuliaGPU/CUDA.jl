export NonReentrantLock, @spinlock, @lock, LazyInitialized

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


"""
    LazyInitialized{T}() do
        # initialization, producing a value of type T
    end

A thread-safe, lazily-initialized wrapper for a value of type `T`. Fetch the value by
calling `getindex`. The constructor is ensured to only be called once from a single thread.

This type is intended for lazy initialization of e.g. global structures, without using
`__init__`. It is similar to protecting accesses using a lock, but is much cheaper.

"""
struct LazyInitialized{T,F}
    # 0: uninitialized
    # 1: initializing
    # 2: initialized
    guard::Threads.Atomic{Int}
    value::Base.RefValue{T}
    constructor::F

    LazyInitialized{T,F}(constructor::F) where {T,F} =
        new(Threads.Atomic{Int}(0), Ref{T}(), constructor)
end
LazyInitialized{T}(constructor::F) where {T,F} = LazyInitialized{T,F}(constructor)

function Base.getindex(x::LazyInitialized; hook=nothing)
    while x.guard[] != 2
        initialize!(x, hook)
    end
    assume(isassigned(x.value)) # to get rid of the check
    x.value[]
end

@noinline function initialize!(x::LazyInitialized, hook::F) where {F}
    status = Threads.atomic_cas!(x.guard, 0, 1)
    if status == 0
        try
          x.value[] = x.constructor()
          x.guard[] = 2
          if hook !== nothing
            hook()
          end
        finally
          x.guard[] = 0
        end
    else
        yield()
    end
    return
end
