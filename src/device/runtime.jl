# CUDAnative run-time library

module Runtime

using ..CUDAnative
using LLVM

struct Function
    def::Core.Function
    return_type::Type
    types::Tuple
    name::String

    Function(def, return_type, types, name=String(typeof(def).name.mt.name)) =
        new(def, return_type, types, name)
end

function Base.getproperty(rt::Function, fld::Symbol)
    # overloaded field accessor to get LLVM types (this only works at run-time,
    # ie. after the Runtime.Function has been constructed)
    if fld == :llvm_types
        LLVMType[convert.(LLVMType, typ) for typ in rt.types]
    elseif fld == :llvm_return_type
        convert.(LLVMType, rt.return_type)
    else
        return getfield(rt, fld)
    end
end


## generic functions

function ptx_report_exception(ex)
    @cuprintf("ERROR: a %s exception occurred during kernel execution\n", ex)
    return
end

const report_exception = Function(ptx_report_exception, Nothing, (Ptr{Cchar},))


end
