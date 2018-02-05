# Device pointer with address space information

#
# Address spaces
#

export
    AS, addrspace

abstract type AddressSpace end

module AS

using CUDAnative
import CUDAnative: AddressSpace

struct Generic  <: AddressSpace end
struct Global   <: AddressSpace end
struct Shared   <: AddressSpace end
struct Constant <: AddressSpace end
struct Local    <: AddressSpace end

end


#
# Device pointer
#

struct DevicePtr{T,A}
    ptr::Ptr{T}

    # inner constructors, fully parameterized
    DevicePtr{T,A}(ptr::Ptr{T}) where {T,A<:AddressSpace} = new(ptr)
end

# outer constructors, partially parameterized
DevicePtr{T}(ptr::Ptr{T}) where {T} = DevicePtr{T,AS.Generic}(ptr)

# outer constructors, non-parameterized
DevicePtr(ptr::Ptr{T})              where {T} = DevicePtr{T,AS.Generic}(ptr)


## getters

Base.pointer(p::DevicePtr) = p.ptr

Base.isnull(p::DevicePtr) = (pointer(p) == C_NULL)
Base.eltype(::Type{<:DevicePtr{T}}) where {T} = T

addrspace(x::DevicePtr) = addrspace(typeof(x))
addrspace(::Type{DevicePtr{T,A}}) where {T,A} = A


## conversions

# between regular and device pointers
## simple conversions disallowed
Base.convert(::Type{Ptr{T}}, p::DevicePtr{T})        where {T} = throw(InexactError(:convert, Ptr{T}, p))
Base.convert(::Type{<:DevicePtr{T}}, p::Ptr{T})      where {T} = throw(InexactError(:convert, DevicePtr{T}, p))
## unsafe ones are allowed
Base.unsafe_convert(::Type{Ptr{T}}, p::DevicePtr{T}) where {T} = pointer(p)

# defer conversions to DevicePtr to unsafe_convert
Base.cconvert(::Type{<:DevicePtr}, x) = x

# between device pointers
Base.convert(::Type{<:DevicePtr}, p::DevicePtr)                         = throw(InexactError(:convert, DevicePtr, p))
Base.convert(::Type{DevicePtr{T,A}}, p::DevicePtr{T,A})   where {T,A}   = p
Base.unsafe_convert(::Type{DevicePtr{T,A}}, p::DevicePtr) where {T,A}   = DevicePtr{T,A}(reinterpret(Ptr{T}, pointer(p)))
## identical addrspaces
Base.convert(::Type{DevicePtr{T,A}}, p::DevicePtr{U,A}) where {T,U,A} = Base.unsafe_convert(DevicePtr{T,A}, p)
## convert to & from generic
Base.convert(::Type{DevicePtr{T,AS.Generic}}, p::DevicePtr)               where {T}     = Base.unsafe_convert(DevicePtr{T,AS.Generic}, p)
Base.convert(::Type{DevicePtr{T,A}}, p::DevicePtr{U,AS.Generic})          where {T,U,A} = Base.unsafe_convert(DevicePtr{T,A}, p)
Base.convert(::Type{DevicePtr{T,AS.Generic}}, p::DevicePtr{T,AS.Generic}) where {T}     = p  # avoid ambiguities
## unspecified, preserve source addrspace
Base.convert(::Type{DevicePtr{T}}, p::DevicePtr{U,A}) where {T,U,A} = Base.unsafe_convert(DevicePtr{T,A}, p)


## limited pointer arithmetic & comparison

Base.:(==)(a::DevicePtr, b::DevicePtr) = pointer(a) == pointer(b) && addrspace(a) == addrspace(b)

Base.isless(x::DevicePtr, y::DevicePtr) = Base.isless(pointer(x), pointer(y))
Base.:(-)(x::DevicePtr, y::DevicePtr)   = pointer(x) - pointer(y)

Base.:(+)(x::DevicePtr{T,A}, y::Integer) where {T,A} = DevicePtr{T,A}(pointer(x) + y)
Base.:(-)(x::DevicePtr{T,A}, y::Integer) where {T,A} = DevicePtr{T,A}(pointer(x) - y)
Base.:(+)(x::Integer, y::DevicePtr) = y + x


## memory operations

Base.convert(::Type{Int}, ::Type{AS.Generic})  = 0
Base.convert(::Type{Int}, ::Type{AS.Global})   = 1
Base.convert(::Type{Int}, ::Type{AS.Shared})   = 3
Base.convert(::Type{Int}, ::Type{AS.Constant}) = 4
Base.convert(::Type{Int}, ::Type{AS.Local})    = 5

@generated function Base.unsafe_load(p::DevicePtr{T,A}, i::Integer=1,
                                     ::Val{align}=Val(1)) where {T,A,align}
    eltyp = convert(LLVMType, T)

    T_int = convert(LLVMType, Int)
    T_ptr = convert(LLVMType, Ptr{T})

    T_actual_ptr = LLVM.PointerType(eltyp)

    # create a function
    param_types = [T_ptr, T_int]
    llvm_f, _ = create_function(eltyp, param_types)

    # generate IR
    Builder(jlctx[]) do builder
        entry = BasicBlock(llvm_f, "entry", jlctx[])
        position!(builder, entry)

        if VERSION >= v"0.7.0-DEV.1704"
            ptr = inttoptr!(builder, parameters(llvm_f)[1], T_actual_ptr)
        else
            ptr = parameters(llvm_f)[1]
        end

        ptr = gep!(builder, ptr, [parameters(llvm_f)[2]])
        ptr_with_as = addrspacecast!(builder, ptr, LLVM.PointerType(eltyp, convert(Int, A)))
        val = load!(builder, ptr_with_as)
        alignment!(val, align)
        ret!(builder, val)
    end

    call_function(llvm_f, T, Tuple{Ptr{T}, Int}, :((pointer(p), Int(i-one(i)))))
end

@generated function Base.unsafe_store!(p::DevicePtr{T,A}, x, i::Integer=1,
                                       ::Val{align}=Val(1)) where {T,A,align}
    eltyp = convert(LLVMType, T)

    T_int = convert(LLVMType, Int)
    T_ptr = convert(LLVMType, Ptr{T})

    T_actual_ptr = LLVM.PointerType(eltyp)

    # create a function
    param_types = [T_ptr, eltyp, T_int]
    llvm_f, _ = create_function(LLVM.VoidType(jlctx[]), param_types)

    # generate IR
    Builder(jlctx[]) do builder
        entry = BasicBlock(llvm_f, "entry", jlctx[])
        position!(builder, entry)

        if VERSION >= v"0.7.0-DEV.1704"
            ptr = inttoptr!(builder, parameters(llvm_f)[1], T_actual_ptr)
        else
            ptr = parameters(llvm_f)[1]
        end

        ptr = gep!(builder, ptr, [parameters(llvm_f)[3]])
        ptr_with_as = addrspacecast!(builder, ptr, LLVM.PointerType(eltyp, convert(Int, A)))
        val = parameters(llvm_f)[2]
        inst = store!(builder, val, ptr_with_as)
        alignment!(inst, align)
        ret!(builder)
    end

    call_function(llvm_f, Cvoid, Tuple{Ptr{T}, T, Int}, :((pointer(p), convert(T,x), Int(i-one(i)))))
end

## loading through the texture cache

# TODO: aren't there more caching options?
#       https://devtalk.nvidia.com/default/topic/938474/8-0-rc-has-new-global-load-intrinsics-with-explicit-cache-modifiers/

const UncachedOperands = Union{Int8,  UInt8,
                               Int16, UInt16,
                               Int32, UInt32,
                               Int64, UInt64,
                               Float32,
                               Float64}

@generated function unsafe_cached_load(p::DevicePtr{T,AS.Global}, i::Integer=1,
                                       ::Val{align}=Val(1)) where
                                      {T<:UncachedOperands,align}
    # NOTE: we can't `ccall(..., llvmcall)`, because
    #       1) Julia passes pointer arguments as plain integers
    #       2) we need to addrspacecast the pointer argument

    eltyp = convert(LLVMType, T)

    T_int = convert(LLVMType, Int)
    T_int32 = LLVM.IntType(32, jlctx[])
    T_ptr = convert(LLVMType, Ptr{T})

    T_actual_ptr = LLVM.PointerType(eltyp)
    T_actual_ptr_as = LLVM.PointerType(eltyp, convert(Int, AS.Global))

    # create a function
    param_types = [T_ptr, T_int]
    llvm_f, _ = create_function(eltyp, param_types)

    # create the intrinsic
    intrinsic_name = let
        class = if isa(eltyp, LLVM.IntegerType)
            :i
        elseif isa(eltyp, LLVM.FloatingPointType)
            :f
        else
            error("Cannot handle $eltyp argument to unsafe_cached_load")
        end
        width = sizeof(T)*8
        typ = Symbol(class, width)
        "llvm.nvvm.ldg.global.$class.$typ.p1$typ"
    end
    mod = LLVM.parent(llvm_f)
    intrinsic_typ = LLVM.FunctionType(eltyp, [T_actual_ptr_as, T_int32])
    intrinsic = LLVM.Function(mod, intrinsic_name, intrinsic_typ)

    # generate IR
    Builder(jlctx[]) do builder
        entry = BasicBlock(llvm_f, "entry", jlctx[])
        position!(builder, entry)

        if VERSION >= v"0.7.0-DEV.1704"
            ptr = inttoptr!(builder, parameters(llvm_f)[1], T_actual_ptr)
        else
            ptr = parameters(llvm_f)[1]
        end

        ptr = gep!(builder, ptr, [parameters(llvm_f)[2]])
        ptr_with_as = addrspacecast!(builder, ptr, T_actual_ptr_as)
        val = call!(builder, intrinsic, [ptr_with_as, ConstantInt(T_int32, align)])
        ret!(builder, val)
    end

    call_function(llvm_f, T, Tuple{Ptr{T}, Int}, :((pointer(p), Int(i-one(i)))))
end

# TODO: extend the recurse/split_invocation infrastructure to work with pointers
# @inline function unsafe_cached_load(p::DevicePtr{T,AS.Global}, args...) where {T}
#     recurse_invocation(unsafe_cached_load, p, args...)
# end
