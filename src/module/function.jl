# Functions in modules

import Base: unsafe_convert

export
    CuFunction


typealias CuFunction_t Ptr{Void}

immutable CuFunction
    handle::CuFunction_t

    "Get a handle to a kernel function in a CUDA module."
    function CuFunction(mod::CuModule, name::String)
        handle_ref = Ref{CuFunction_t}()
        @apicall(:cuModuleGetFunction, (Ptr{CuFunction_t}, CuModule_t, Ptr{Cchar}),
                                      handle_ref, mod.handle, name)
        new(handle_ref[])
    end
end

unsafe_convert(::Type{CuFunction_t}, fun::CuFunction) = fun.handle
