# Shared Memory (part of B.2)

export @cuStaticSharedMem, @cuDynamicSharedMem, CuStaticSharedArray, CuDynamicSharedArray, CuDistributedSharedArray

"""
    CuStaticSharedArray(T::Type, dims) -> CuDeviceArray{T,N,AS.Shared}

Get an array of type `T` and dimensions `dims` (either an integer length or tuple shape)
pointing to a statically-allocated piece of shared memory. The type should be statically
inferable and the dimensions should be constant, or an error will be thrown and the
generator function will be called dynamically.
"""
@inline function CuStaticSharedArray(::Type{T}, dims::Tuple) where {T}
    N = length(dims)
    len = prod(dims)
    # NOTE: this relies on const-prop to forward the literal length to the generator.
    #       maybe we should include the size in the type, like StaticArrays does?
    ptr = emit_shmem(T, Val(len))
    CuDeviceArray{T,N,AS.Shared}(ptr, dims)
end
CuStaticSharedArray(::Type{T}, len::Integer) where {T} = CuStaticSharedArray(T, (len,))

macro cuStaticSharedMem(T, dims)
    Base.depwarn("@cuStaticSharedMem is deprecated, please use the CuStaticSharedArray function", :CuStaticSharedArray)
    quote
        CuStaticSharedArray($(esc(T)), $(esc(dims)))
    end
end

"""
    CuDynamicSharedArray(T::Type, dims, offset::Integer=0) -> CuDeviceArray{T,N,AS.Shared}

Get an array of type `T` and dimensions `dims` (either an integer length or tuple shape)
pointing to a dynamically-allocated piece of shared memory. The type should be statically
inferable or an error will be thrown and the generator function will be called dynamically.

Note that the amount of dynamic shared memory needs to specified when launching the kernel.

Optionally, an offset parameter indicating how many bytes to add to the base shared memory
pointer can be specified. This is useful when dealing with a heterogeneous buffer of dynamic
shared memory; in the case of a homogeneous multi-part buffer it is preferred to use `view`.
"""
@inline function CuDynamicSharedArray(::Type{T}, dims::Tuple, offset) where {T}
    N = length(dims)
    @boundscheck begin
        len = prod(dims)
        sz = len*sizeof(T)
        if !isbitstype(T)
            sz += len
        end
        if offset+sz > dynamic_smem_size()
            throw(BoundsError())
        end
    end
    ptr = emit_shmem(T) + offset
    CuDeviceArray{T,N,AS.Shared}(ptr, dims)
end
Base.@propagate_inbounds CuDynamicSharedArray(::Type{T}, len::Integer, offset) where {T} =
    CuDynamicSharedArray(T, (len,), offset)
# Default argument-generated methods do not propagate inboundsness
Base.@propagate_inbounds CuDynamicSharedArray(::Type{T}, dims) where {T} =
    CuDynamicSharedArray(T, dims, 0)

macro cuDynamicSharedMem(T, dims, offset=0)
    Base.depwarn("@cuDynamicSharedMem is deprecated, please use the CuDynamicSharedArray function", :CuStaticSharedArray)
    quote
        CuDynamicSharedArray($(esc(T)), $(esc(dims)), $(esc(offset)))
    end
end

@device_function dynamic_smem_size() =
    @asmcall("mov.u32 \$0, %dynamic_smem_size;", "=r", true, UInt32, Tuple{})

@inline function CuDistributedSharedArray(shared_array::CuDeviceArray{T,N,AS.Shared}, blockidx::Integer) where {T,N}
    # Distributed shared memory has address space 7 (SharedCluster).
    # This is only supported in LLVM >= 21 which we can't yet use with
    # Julia. We therefore need to map it to address space 0 (Generic).
    #
    # We should change this to be address space 7 (SharedCluster) if
    # we're using LLVM >=21.

    ptr = map_shared_rank(shared_array.ptr, blockidx)
    CuDeviceArray{T,N,AS.Generic}(ptr, shared_array.dims, shared_array.maxsize)
end

@device_function @inline function map_shared_rank(ptr_shared::LLVMPtr{T,AS.Shared}, rank::Integer) where {T}
    require_sm_90()
    # This requires LLVM >=20 (i.e. Julia >= 1.13)
    ptr7 = @asmcall(
        "mapa.shared::cluster.u64 \$0, \$1, \$2;",
        "=l,l,r",
        LLVMPtr{T,AS.SharedCluster},
        Tuple{Core.LLVMPtr{T,AS.Shared}, Int32},
        ptr_shared, Int32(rank - 1i32),
    )
    ptr0 = @asmcall(
        "cvta.shared::cluster.u64 \$0, \$1;",
        "=l,l",
        LLVMPtr{T,AS.Generic},
        Tuple{Core.LLVMPtr{T,AS.SharedCluster}},
        ptr7,
    )
    return ptr0
end

# get a pointer to shared memory, with known (static) or zero length (dynamic shared memory)
@llvmgenerated builder function emit_shmem(::Type{T},
                                           ::Val{len}=Val(0))::LLVMPtr{T,AS.Shared} where {T,len}
    T_int8 = LLVM.Int8Type()
    T_ptr = convert(LLVMType, LLVMPtr{T,AS.Shared})

    # determine the array size
    # TODO: assert that allocatedinline(T) (or it won't have a layout)
    sz = len*sizeof(T)
    if !isbitstype(T)
        sz += len
    end

    # create the global variable
    # NOTE: this variable can't have T as element type, because it may be a boxed type
    #       when we're dealing with a union isbits array (e.g. `Union{Missing,Int}`)
    gv_typ = LLVM.ArrayType(T_int8, sz)
    gv = GlobalVariable(current_module(builder), gv_typ, "shmem", AS.Shared)
    if len > 0
        # static shared memory should be demoted to local variables, whenever possible.
        # this is done by the NVPTX ASM printer:
        # > Find out if a global variable can be demoted to local scope.
        # > Currently, this is valid for CUDA shared variables, which have local
        # > scope and global lifetime. So the conditions to check are :
        # > 1. Is the global variable in shared address space?
        # > 2. Does it have internal linkage?
        # > 3. Is the global variable referenced only in one function?
        gv.linkage = LLVM.Linkage.Internal
        gv.initializer = null(gv_typ)
    end
    # by requesting a larger-than-datatype alignment, we might be able to vectorize.
    # we pick 32 bytes here, since WMMA instructions require 32-byte alignment.
    # TODO: Make the alignment configurable
    align = 32
    if isbitstype(T)
        align = max(align, Base.datatype_alignment(T))
    else # isbitsunion etc
        for typ in Base.uniontypes(T)
            if typ.layout != C_NULL
                align = max(align, Base.datatype_alignment(typ))
            end
        end
    end
    gv.alignment = align

    ptr = gep!(builder, gv_typ, gv, [ConstantInt(0), ConstantInt(0)])
    bitcast!(builder, ptr, T_ptr)
end


# Dynamic Global Memory Allocation and Operations (B.21)

export malloc

@llvmgenerated builder function malloc(sz::Csize_t)::Ptr{Cvoid}
    T_pint8 = LLVM.PointerType(LLVM.Int8Type())
    T_size = convert(LLVMType, Csize_t)
    T_ptr = convert(LLVMType, Ptr{Cvoid})

    # get the intrinsic
    # NOTE: LLVM doesn't have void*, Clang uses i8* for malloc too
    intr_typ = LLVM.FunctionType(T_pint8, [T_size])
    intr = LLVM.Function(current_module(builder), "malloc", intr_typ)
    # should we attach some metadata here? julia.gc_alloc_obj has the following:
    #let attrs = intr.function_attributes
    #    AllocSizeNumElemsNotPresent = reinterpret(Cuint, Cint(-1))
    #    packed_allocsize = Int64(1) << 32 | AllocSizeNumElemsNotPresent
    #    push!(attrs, EnumAttribute(:allocsize, packed_allocsize))
    #end
    #let attrs = intr.return_attributes
    #    push!(attrs, EnumAttribute(:noalias))
    #    push!(attrs, EnumAttribute(:nonnull))
    #end

    ptr = call!(builder, intr_typ, intr, [sz])
    pointercast!(builder, ptr, T_ptr)
end
