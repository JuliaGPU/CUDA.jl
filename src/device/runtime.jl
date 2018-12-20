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
        convert.(LLVMType, rt.return_type)
    else
        return getfield(rt, field)
    end
end


## exception handling

function ptx_report_exception(ex)
    @cuprintf("ERROR: a %s exception occurred during kernel execution\n", ex)
    return
end

const report_exception = instantiate(ptx_report_exception, Nothing, (Ptr{Cchar},))


end
