export LazyInitialized

"""
    LazyInitialized{T}()

A thread-safe, lazily-initialized wrapper for a value of type `T`. Initialize and fetch the
value by calling `get!`. The constructor is ensured to only be called once.

This type is intended for lazy initialization of e.g. global structures, without using
`__init__`. It is similar to protecting accesses using a lock, but is much cheaper.

"""
struct LazyInitialized{T}
    # 0: uninitialized
    # 1: initializing
    # 2: initialized
    guard::Threads.Atomic{Int}
    value::Base.RefValue{T}

    LazyInitialized{T}() where {T} =
        new(Threads.Atomic{Int}(0), Ref{T}())
end

@inline function Base.get!(constructor, x::LazyInitialized; hook=nothing)
    while x.guard[] != 2
        initialize!(x, constructor, hook)
    end
    assume(isassigned(x.value)) # to get rid of the check
    x.value[]
end

@noinline function initialize!(x::LazyInitialized, constructor::F1, hook::F2) where {F1, F2}
    status = Threads.atomic_cas!(x.guard, 0, 1)
    if status == 0
        try
          x.value[] = constructor()
          x.guard[] = 2
        catch
          x.guard[] = 0
          rethrow()
        end

        if hook !== nothing
          hook()
        end
    else
        yield()
    end
    return
end
