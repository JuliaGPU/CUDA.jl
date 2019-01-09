# CUDAnative run-time library
#
# This module defines method instances that will be compiled into a device-specific image
# and will be available to the CUDAnative compiler to call after Julia has generated code.

module Runtime

using ..CUDAnative
using LLVM

struct MethodInstance
    def::Function
    return_type::Type
    types::Tuple
    name::String
end

instantiate(def, return_type, types, name=String(typeof(def).name.mt.name)) =
    MethodInstance(def, return_type, types, name)

function Base.getproperty(rt::MethodInstance, field::Symbol)
    # overloaded field accessor to get LLVM types (this only works at run-time,
    # ie. after the method instance has been constructed and precompiled)
    if field == :llvm_types
        LLVMType[convert.(LLVMType, typ) for typ in rt.types]
    elseif field == :llvm_return_type
        convert.(LLVMType, rt.return_type, true)
    else
        return getfield(rt, field)
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

function ptx_alloc_obj(sz::Csize_t)
    ptr = malloc(sz)
    return unsafe_pointer_to_objref(ptr) # this returns a tracked pointer
end

const alloc_obj = instantiate(ptx_alloc_obj, Any, (Csize_t,))

# sadly, due to how our runtime is constructed, we create the LLVM function prototype by
# only looking at the Julia types. When doing `convert(LLVMType, Any)` (aka.
# `julia_type_to_llvm`), we get a `jl_value_t*` that has lost its `addrspace(10)` tracking
# information. for now, we just call malloc directly and reconstruct the addrspace manually

const malloc = instantiate(CUDAnative.malloc, Ptr{Cvoid}, (Csize_t,))


end
