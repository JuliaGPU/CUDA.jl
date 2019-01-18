# CUDAnative run-time library
#
# This module defines method instances that will be compiled into a device-specific image
# and will be available to the CUDAnative compiler to call after Julia has generated code.

module Runtime

using ..CUDAnative
using LLVM
using LLVM.Interop


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

function bounds_error_unboxed_int(data, vt, i)
    @cuprintf("ERROR: a bounds error occurred during kernel execution.")
    # FIXME: have this call emit_exception somehow
    return
end

compile(bounds_error_unboxed_int, Nothing, (Ptr{Cvoid}, Any, Csize_t);
        llvm_name="jl_bounds_error_unboxed_int")


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


## boxing and unboxing

const tag_type = UInt
const tag_size = sizeof(tag_type)

const gc_bits = 0x3 # FIXME

# get the type tag of a type at run-time
@generated function type_tag(::Val{type_name}) where type_name
    T_tag = convert(LLVMType, tag_type)
    T_ptag = LLVM.PointerType(T_tag)

    T_pjlvalue = convert(LLVMType, Any, true)

    # create function
    llvm_f, _ = create_function(T_tag)
    mod = LLVM.parent(llvm_f)

    # this isn't really a function, but we abuse it to get the JIT to resolve the address
    typ = LLVM.Function(mod, "jl_" * String(type_name) * "_type",
                        LLVM.FunctionType(T_pjlvalue))

    # generate IR
    Builder(JuliaContext()) do builder
        entry = BasicBlock(llvm_f, "entry", JuliaContext())
        position!(builder, entry)

        typ_var = bitcast!(builder, typ, T_ptag)

        tag = load!(builder, typ_var)

        ret!(builder, tag)
    end

    call_function(llvm_f, tag_type)
end

# we use `jl_value_ptr`, a Julia pseudo-intrinsic that can be used to box and unbox values

@generated function box(val, ::Val{type_name}) where type_name
    sz = sizeof(val)
    allocsz = sz + tag_size

    # type-tags are ephemeral, so look them up at run time
    #tag = unsafe_load(convert(Ptr{tag_type}, type_name))
    tag = :( type_tag(Val(type_name)) )

    quote
        Base.@_inline_meta

        ptr = malloc($(Csize_t(allocsz)))

        # store the type tag
        ptr = convert(Ptr{tag_type}, ptr)
        Core.Intrinsics.pointerset(ptr, $tag | $gc_bits, #=index=# 1, #=align=# $tag_size)

        # store the value
        ptr = convert(Ptr{$val}, ptr+tag_size)
        Core.Intrinsics.pointerset(ptr, val, #=index=# 1, #=align=# $sz)

        unsafe_pointer_to_objref(ptr)
    end
end

@inline function unbox(obj, ::Type{T}) where T
    ptr = ccall(:jl_value_ptr, Ptr{Cvoid}, (Any,), obj)

    # load the value
    ptr = convert(Ptr{T}, ptr)
    Core.Intrinsics.pointerref(ptr, #=index=# 1, #=align=# sizeof(T))
end

# generate functions functions that exist in the Julia runtime (see julia/src/datatype.c)
for (T, t) in [Int8   => :int8,  Int16  => :int16,  Int32  => :int32,  Int64  => :int64,
               UInt8  => :uint8, UInt16 => :uint16, UInt32 => :uint32, UInt64 => :uint64]
    box_fn   = Symbol("box_$t")
    unbox_fn = Symbol("unbox_$t")
    @eval begin
        $box_fn(val)   = box($T(val), Val($(QuoteNode(t))))
        $unbox_fn(obj) = unbox(obj, $T)

        compile($box_fn, Any, ($T,), T_prjlvalue; llvm_name=$"jl_$box_fn")
        compile($unbox_fn, $T, (Any,); llvm_name=$"jl_$unbox_fn")
    end
end


end
