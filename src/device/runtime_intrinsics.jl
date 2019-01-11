# CUDAnative run-time library
#
# This module defines method instances that will be compiled into a device-specific image
# and will be available to the CUDAnative compiler to call after Julia has generated code.

module Runtime

using ..CUDAnative
using LLVM


## representation of a runtime method instance

struct RuntimeMethodInstance
    def::Function

    return_type::Type
    types::Tuple
    name::Symbol

    # LLVM types cannot be cached, so we can't put them in the runtime method instance.
    # the actual types are constructed upon accessing them, based on a sentinel value:
    #  - nothing: construct the LLVM type based on its Julia counterparts
    #  - function: call this generator to get the type (when more control is needed)
    llvm_return_type::Union{Nothing, Function}
    llvm_types::Union{Nothing, Function}
    llvm_name::String
end

function Base.getproperty(rt::RuntimeMethodInstance, field::Symbol)
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

const methods = Dict{Symbol,RuntimeMethodInstance}()
get(name::Symbol) = methods[name]

# Register a Julia function `def` as a runtime library function identified by `name`. The
# function will be compiled upon first use for argument types `types` and should return
# `return_type`. Use `Runtime.get(name)` to get a reference to this method instance.
#
# The corresponding LLVM types `llvm_types` and `llvm_return_type` will be deduced from
# their Julia counterparts. To influence that conversion, pass a callable object instead;
# this object will be evaluated at run-time and the returned value will be used instead.
#
# When generating multiple runtime functions from a single definition, make sure to specify
# different values for `name`. The LLVM function name will be deduced from that name, but
# you can always specify `llvm_name` to influence that. Never use an LLVM name that starts
# with `julia_` or the function might clash with other compiled functions.
function compile(def, return_type, types, llvm_return_type=nothing, llvm_types=nothing;
                 name=typeof(def).name.mt.name, llvm_name="ptx_$name")
    meth = RuntimeMethodInstance(def,
                                 return_type, types, name,
                                 llvm_return_type, llvm_types, llvm_name)
    if haskey(methods, name)
        error("Runtime function $name has already been registered!")
    end
    methods[name] = meth
    meth
end


## exception handling

function report_exception(ex)
    @cuprintf("""
        ERROR: a %s exception occurred during kernel execution.
               Run Julia on debug level 2 for device stack traces.
        """, ex)
    return
end

compile(report_exception, Nothing, (Ptr{Cchar},))

function report_exception_name(ex)
    @cuprintf("""
        ERROR: a %s exception occurred during kernel execution.
        Stacktrace:
        """, ex)
    return
end

function report_exception_frame(idx, func, file, line)
    @cuprintf(" [%i] %s at %s:%i\n", idx, func, file, line)
    return
end

compile(report_exception_frame, Nothing, (Cint, Ptr{Cchar}, Ptr{Cchar}, Cint))
compile(report_exception_name, Nothing, (Ptr{Cchar},))


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

function gc_pool_alloc(sz::Csize_t)
    ptr = malloc(sz)
    return unsafe_pointer_to_objref(ptr)
end

compile(gc_pool_alloc, Any, (Csize_t,), T_prjlvalue)


end
