# CUDAnative run-time library
#
# This module defines method instances that will be compiled into a device-specific image
# and will be available to the CUDAnative compiler to call after Julia has generated code.

module Runtime

using ..CUDAnative
using LLVM

struct MethodInstance
    def::Function
    name::String

    return_type::Type
    types::Tuple

    # LLVM types cannot be cached, so we can't put them in the method instance.
    # the actual types are constructed upon accessing them based on a sentinel value:
    #  - nothing: construct the LLVM type based on its Julia counterparts
    #  - function: call this generator to get the type (when more control is needed)
    llvm_return_type::Union{Nothing, Function}
    llvm_types::Union{Nothing, Function}
end

instantiate(def, return_type, types, llvm_return_type=nothing, llvm_types=nothing;
            name=String(typeof(def).name.mt.name)) =
    MethodInstance(def, name, return_type, types, llvm_return_type, llvm_types)

function Base.getproperty(rt::MethodInstance, field::Symbol)
    value = getfield(rt, field)
    if field == :llvm_types
        if value == nothing
            LLVMType[convert.(LLVMType, typ) for typ in rt.types]
        else
            value()
        end
    elseif field == :llvm_return_type
        if value == nothing
            convert.(LLVMType, rt.return_type)
        else
            value()
        end
    else
        return value
    end
end


## exception handling

function ptx_report_exception(ex)
    @cuprintf("""
        ERROR: a %s exception occurred during kernel execution.
               Run Julia on debug level 2 for device stack traces.
        """, ex)
    return
end

const report_exception = instantiate(ptx_report_exception, Nothing, (Ptr{Cchar},))

function ptx_report_exception_name(ex)
    @cuprintf("""
        ERROR: a %s exception occurred during kernel execution.
        Stacktrace:
        """, ex)
    return
end

const report_exception_name = instantiate(ptx_report_exception_name, Nothing, (Ptr{Cchar},))

function ptx_report_exception_frame(idx, func, file, line)
    @cuprintf(" [%i] %s at %s:%i\n", idx, func, file, line)
    return
end

const report_exception_frame = instantiate(ptx_report_exception_frame, Nothing,
                                           (Cint, Ptr{Cchar}, Ptr{Cchar}, Cint))


## GC

@enum AddressSpace begin
    Generic         = 1
    Tracked         = 10
    Derived         = 11
    CalleeRooted    = 12
    Loaded          = 13
end

# LLVM type of a tracked pointer
function T_prjlvalue()
    T_pjlvalue = convert(LLVMType, Any, true)
    LLVM.PointerType(eltype(T_pjlvalue), Tracked)
end

function ptx_alloc_obj(sz::Csize_t)
    ptr = malloc(sz)
    return unsafe_pointer_to_objref(ptr)
end

const alloc_obj = instantiate(ptx_alloc_obj, Any, (Csize_t,), T_prjlvalue)


end
