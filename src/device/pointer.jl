# Pointers with address space information

#
# Address spaces
#

export AS, addrspace

abstract type AddressSpace end

module AS

import ..AddressSpace

struct Generic  <: AddressSpace end
struct Global   <: AddressSpace end
struct Shared   <: AddressSpace end
struct Constant <: AddressSpace end
struct Local    <: AddressSpace end

end


#
# Device pointer
#

"""
    DevicePtr{T,A}

A memory address that refers to data of type `T` that is accessible from the GPU. It is the
on-device counterpart of `CUDAdrv.CuPtr`, additionally keeping track of the address space
`A` where the data resides (shared, global, constant, etc). This information is used to
provide optimized implementations of operations such as `unsafe_load` and `unsafe_store!.`
"""
DevicePtr

if sizeof(Ptr{Cvoid}) == 8
    primitive type DevicePtr{T,A} 64 end
else
    primitive type DevicePtr{T,A} 32 end
end

# constructors
DevicePtr{T,A}(x::Union{Int,UInt,CuPtr,DevicePtr}) where {T,A<:AddressSpace} = Base.bitcast(DevicePtr{T,A}, x)
DevicePtr{T}(ptr::CuPtr{T}) where {T} = DevicePtr{T,AS.Generic}(ptr)
DevicePtr(ptr::CuPtr{T}) where {T} = DevicePtr{T,AS.Generic}(ptr)


## getters

Base.eltype(::Type{<:DevicePtr{T}}) where {T} = T

addrspace(x::DevicePtr) = addrspace(typeof(x))
addrspace(::Type{DevicePtr{T,A}}) where {T,A} = A


## conversions

# to and from integers
## pointer to integer
Base.convert(::Type{T}, x::DevicePtr) where {T<:Integer} = T(UInt(x))
## integer to pointer
Base.convert(::Type{DevicePtr{T,A}}, x::Union{Int,UInt}) where {T,A<:AddressSpace} = DevicePtr{T,A}(x)
Int(x::DevicePtr)  = Base.bitcast(Int, x)
UInt(x::DevicePtr) = Base.bitcast(UInt, x)

# between host and device pointers
Base.convert(::Type{CuPtr{T}},  p::DevicePtr)  where {T}                 = Base.bitcast(CuPtr{T}, p)
Base.convert(::Type{DevicePtr{T,A}}, p::CuPtr) where {T,A<:AddressSpace} = Base.bitcast(DevicePtr{T,A}, p)
Base.convert(::Type{DevicePtr{T}}, p::CuPtr)   where {T}                 = Base.bitcast(DevicePtr{T,AS.Generic}, p)

# between device pointers
Base.convert(::Type{<:DevicePtr}, p::DevicePtr)                         = throw(ArgumentError("cannot convert between incompatible device pointer types"))
Base.convert(::Type{DevicePtr{T,A}}, p::DevicePtr{T,A})   where {T,A}   = p
Base.unsafe_convert(::Type{DevicePtr{T,A}}, p::DevicePtr) where {T,A}   = Base.bitcast(DevicePtr{T,A}, p)
## identical addrspaces
Base.convert(::Type{DevicePtr{T,A}}, p::DevicePtr{U,A}) where {T,U,A} = Base.unsafe_convert(DevicePtr{T,A}, p)
## convert to & from generic
Base.convert(::Type{DevicePtr{T,AS.Generic}}, p::DevicePtr)               where {T}     = Base.unsafe_convert(DevicePtr{T,AS.Generic}, p)
Base.convert(::Type{DevicePtr{T,A}}, p::DevicePtr{U,AS.Generic})          where {T,U,A} = Base.unsafe_convert(DevicePtr{T,A}, p)
Base.convert(::Type{DevicePtr{T,AS.Generic}}, p::DevicePtr{T,AS.Generic}) where {T}     = p  # avoid ambiguities
## unspecified, preserve source addrspace
Base.convert(::Type{DevicePtr{T}}, p::DevicePtr{U,A}) where {T,U,A} = Base.unsafe_convert(DevicePtr{T,A}, p)

# defer conversions to DevicePtr to unsafe_convert
Base.cconvert(::Type{<:DevicePtr}, x) = x


## limited pointer arithmetic & comparison

isequal(x::DevicePtr, y::DevicePtr) = (x === y) && addrspace(x) == addrspace(y)
isless(x::DevicePtr{T,A}, y::DevicePtr{T,A}) where {T,A<:AddressSpace} = x < y

Base.:(==)(x::DevicePtr, y::DevicePtr) = UInt(x) == UInt(y) && addrspace(x) == addrspace(y)
Base.:(<)(x::DevicePtr,  y::DevicePtr) = UInt(x) < UInt(y)
Base.:(-)(x::DevicePtr,  y::DevicePtr) = UInt(x) - UInt(y)

Base.:(+)(x::DevicePtr, y::Integer) = oftype(x, Base.add_ptr(UInt(x), (y % UInt) % UInt))
Base.:(-)(x::DevicePtr, y::Integer) = oftype(x, Base.sub_ptr(UInt(x), (y % UInt) % UInt))
Base.:(+)(x::Integer, y::DevicePtr) = y + x



## memory operations

Base.convert(::Type{Int}, ::Type{AS.Generic})  = 0
Base.convert(::Type{Int}, ::Type{AS.Global})   = 1
Base.convert(::Type{Int}, ::Type{AS.Shared})   = 3
Base.convert(::Type{Int}, ::Type{AS.Constant}) = 4
Base.convert(::Type{Int}, ::Type{AS.Local})    = 5

function tbaa_make_child(name::String, constant::Bool=false; ctx::LLVM.Context=JuliaContext())
    tbaa_root = MDNode([MDString("ptxtbaa", ctx)], ctx)
    tbaa_struct_type =
        MDNode([MDString("ptxtbaa_$name", ctx),
                tbaa_root,
                LLVM.ConstantInt(0, ctx)], ctx)
    tbaa_access_tag =
        MDNode([tbaa_struct_type,
                tbaa_struct_type,
                LLVM.ConstantInt(0, ctx),
                LLVM.ConstantInt(constant ? 1 : 0, ctx)], ctx)

    return tbaa_access_tag
end

tbaa_addrspace(as::Type{<:AddressSpace}) = tbaa_make_child(lowercase(String(as.name.name)))

@generated function Base.unsafe_load(p::DevicePtr{T,A}, i::Integer=1,
                                     ::Val{align}=Val(1)) where {T,A,align}
    eltyp = convert(LLVMType, T)

    T_int = convert(LLVMType, Int)
    T_ptr = convert(LLVMType, DevicePtr{T,A})

    T_actual_ptr = LLVM.PointerType(eltyp)

    # create a function
    param_types = [T_ptr, T_int]
    llvm_f, _ = create_function(eltyp, param_types)

    # generate IR
    Builder(JuliaContext()) do builder
        entry = BasicBlock(llvm_f, "entry", JuliaContext())
        position!(builder, entry)

        ptr = inttoptr!(builder, parameters(llvm_f)[1], T_actual_ptr)

        ptr = gep!(builder, ptr, [parameters(llvm_f)[2]])
        ptr_with_as = addrspacecast!(builder, ptr, LLVM.PointerType(eltyp, convert(Int, A)))
        ld = load!(builder, ptr_with_as)

        if A != AS.Generic
            metadata(ld)[LLVM.MD_tbaa] = tbaa_addrspace(A)
        end
        alignment!(ld, align)

        ret!(builder, ld)
    end

    call_function(llvm_f, T, Tuple{DevicePtr{T,A}, Int}, :((p, Int(i-one(i)))))
end

@generated function Base.unsafe_store!(p::DevicePtr{T,A}, x, i::Integer=1,
                                       ::Val{align}=Val(1)) where {T,A,align}
    eltyp = convert(LLVMType, T)

    T_int = convert(LLVMType, Int)
    T_ptr = convert(LLVMType, DevicePtr{T,A})

    T_actual_ptr = LLVM.PointerType(eltyp)

    # create a function
    param_types = [T_ptr, eltyp, T_int]
    llvm_f, _ = create_function(LLVM.VoidType(JuliaContext()), param_types)

    # generate IR
    Builder(JuliaContext()) do builder
        entry = BasicBlock(llvm_f, "entry", JuliaContext())
        position!(builder, entry)

        ptr = inttoptr!(builder, parameters(llvm_f)[1], T_actual_ptr)

        ptr = gep!(builder, ptr, [parameters(llvm_f)[3]])
        ptr_with_as = addrspacecast!(builder, ptr, LLVM.PointerType(eltyp, convert(Int, A)))
        val = parameters(llvm_f)[2]
        st = store!(builder, val, ptr_with_as)

        if A != AS.Generic
            metadata(st)[LLVM.MD_tbaa] = tbaa_addrspace(A)
        end
        alignment!(st, align)

        ret!(builder)
    end

    call_function(llvm_f, Cvoid, Tuple{DevicePtr{T,A}, T, Int},
                  :((p, convert(T,x), Int(i-one(i)))))
end

## loading through the texture cache

export unsafe_cached_load

# NOTE: CUDA 8.0 supports more caching modifiers, but those aren't supported by LLVM yet

# TODO: this functionality should throw <sm_32

# operand types supported by llvm.nvvm.ldg.global
const CachedLoadOperands = Union{UInt8, UInt16, UInt32, UInt64,
                                 Int8, Int16, Int32, Int64,
                                 Float32, Float64}

# containing DevicePtr types
const CachedLoadPointers = Union{Tuple(DevicePtr{T,AS.Global}
                                 for T in Base.uniontypes(CachedLoadOperands))...}

@generated function unsafe_cached_load(p::DevicePtr{T,AS.Global}, i::Integer=1,
                                       ::Val{align}=Val(1)) where
                                      {T<:CachedLoadOperands,align}
    # NOTE: we can't `ccall(..., llvmcall)`, because
    #       1) Julia passes pointer arguments as plain integers
    #       2) we need to addrspacecast the pointer argument

    eltyp = convert(LLVMType, T)

    T_int = convert(LLVMType, Int)
    T_int32 = LLVM.Int32Type(JuliaContext())
    T_ptr = convert(LLVMType, DevicePtr{T,AS.Global})

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
    Builder(JuliaContext()) do builder
        entry = BasicBlock(llvm_f, "entry", JuliaContext())
        position!(builder, entry)

        ptr = inttoptr!(builder, parameters(llvm_f)[1], T_actual_ptr)

        ptr = gep!(builder, ptr, [parameters(llvm_f)[2]])
        ptr_with_as = addrspacecast!(builder, ptr, T_actual_ptr_as)
        ld = call!(builder, intrinsic,
                   [ptr_with_as, ConstantInt(Int32(align), JuliaContext())])

        metadata(ld)[LLVM.MD_tbaa] = tbaa_addrspace(AS.Global)

        ret!(builder, ld)
    end

    call_function(llvm_f, T, Tuple{DevicePtr{T,AS.Global}, Int}, :((p, Int(i-one(i)))))
end

@inline unsafe_cached_load(p::DevicePtr{T,AS.Global}, i::Integer=1, args...) where {T} =
    recurse_pointer_invocation(unsafe_cached_load, p+sizeof(T)*Int(i-one(i)),
                               CachedLoadPointers, 1, args...)
