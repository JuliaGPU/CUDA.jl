# Functions in modules

export
    CuFunction


"""
    CuFunction(mod::CuModule, name::String)

Acquires a function handle from a named function in a module.
"""
struct CuFunction
    handle::CUfunction
    mod::CuModule

    "Get a handle to a kernel function in a CUDA module."
    function CuFunction(mod::CuModule, name::String)
        handle_ref = Ref{CUfunction}()
        cuModuleGetFunction(handle_ref, mod, name)
        new(handle_ref[], mod)
    end
end

Base.unsafe_convert(::Type{CUfunction}, fun::CuFunction) = fun.handle

Base.:(==)(a::CuFunction, b::CuFunction) = a.handle == b.handle
Base.hash(fun::CuFunction, h::UInt) = hash(mod.handle, h)
