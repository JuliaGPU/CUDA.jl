# Functions in modules

export
    CuFunction


const CuFunction_t = Ptr{Void}

"""
    CuFunction(mod::CuModule, name::String)

Acquires a function handle from a named function in a module.
"""
struct CuFunction
    handle::CuFunction_t
    mod::CuModule

    "Get a handle to a kernel function in a CUDA module."
    function CuFunction(mod::CuModule, name::String)
        handle_ref = Ref{CuFunction_t}()
        @apicall(:cuModuleGetFunction, (Ptr{CuFunction_t}, CuModule_t, Ptr{Cchar}),
                                       handle_ref, mod, name)
        new(handle_ref[], mod)
    end
end

Base.unsafe_convert(::Type{CuFunction_t}, fun::CuFunction) = fun.handle

Base.:(==)(a::CuFunction, b::CuFunction) = a.handle == b.handle
Base.hash(fun::CuFunction, h::UInt) = hash(mod.handle, h)
