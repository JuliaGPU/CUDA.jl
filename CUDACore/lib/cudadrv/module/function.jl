# Functions in modules

export CuFunction


"""
    CuFunction(mod::CuModule, name::String)

Acquires a function handle from a named function in a module.
"""
mutable struct CuFunction
    const handle::CUfunction
    const mod::CuModule

    # queried for every launch, so cached; see `max_threads_per_block` and
    # `launch_configuration`
    Base.@atomic max_threads::Int    # 0 until queried
    Base.@atomic launch_config::Any  # `nothing` or a `CachedLaunchConfiguration`

    "Get a handle to a kernel function in a CUDA module."
    function CuFunction(mod::CuModule, name::String)
        handle_ref = Ref{CUfunction}()
        cuModuleGetFunction(handle_ref, mod, name)
        new(handle_ref[], mod, 0, nothing)
    end
end

Base.unsafe_convert(::Type{CUfunction}, fun::CuFunction) = fun.handle

Base.:(==)(a::CuFunction, b::CuFunction) = a.handle == b.handle
Base.hash(fun::CuFunction, h::UInt) = hash(fun.handle, h)
