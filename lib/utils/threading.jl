export LazyInitialized

"""
    LazyInitialized{T}()

A thread-safe, lazily-initialized wrapper for a value of type `T`. Initialize and fetch the
value by calling `get!`. The constructor is ensured to only be called once.

This type is intended for lazy initialization of e.g. global structures, without using
`__init__`. It is similar to protecting accesses using a lock, but is much cheaper.

"""
struct LazyInitialized{T,F}
    # 0: uninitialized
    # 1: initializing
    # 2: initialized
    guard::Threads.Atomic{Int}
    value::Base.RefValue{T}
    # XXX: use Base.ThreadSynchronizer instead?

    validator::F
end

LazyInitialized{T}(validator=nothing) where {T} =
    LazyInitialized{T,typeof(validator)}(Threads.Atomic{Int}(0), Ref{T}(), validator)

@inline function Base.get!(constructor::Base.Callable, x::LazyInitialized)
    while x.guard[] != 2
        initialize!(x, constructor)
    end
    assume(isassigned(x.value)) # to get rid of the check
    val = x.value[]

    # check if the value is still valid
    if x.validator !== nothing && !x.validator(val)
        Threads.atomic_cas!(x.guard, 2, 0)
        while x.guard[] != 2
            initialize!(x, constructor)
        end
        assume(isassigned(x.value))
        val = x.value[]
    end

    return val
end

@noinline function initialize!(x::LazyInitialized{T}, constructor::F) where {T, F}
    status = Threads.atomic_cas!(x.guard, 0, 1)
    if status == 0
        try
          x.value[] = constructor()::T
          x.guard[] = 2
        catch
          x.guard[] = 0
          rethrow()
        end
    else
        ccall(:jl_cpu_suspend, Cvoid, ())
        # Temporary solution before we have gc transition support in codegen.
        ccall(:jl_gc_safepoint, Cvoid, ())
    end
    return
end
