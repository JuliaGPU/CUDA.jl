# Shared Memory (part of B.2)

export @cuStaticSharedMem, @cuDynamicSharedMem, CuStaticSharedArray, CuDynamicSharedArray

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
    # XXX: 4GB ought to be enough shared memory for anybody
    CuDeviceArray{T,N,AS.Shared,Int32}(ptr, dims)
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
    # XXX: 4GB ought to be enough shared memory for anybody
    CuDeviceArray{T,N,AS.Shared,Int32}(ptr, dims)
end
Base.@propagate_inbounds CuDynamicSharedArray(::Type{T}, len::Integer, offset) where {T} =
    CuDynamicSharedArray(T, (len,), offset)
# XXX: default argument-generated methods do not propagate inboundsness
Base.@propagate_inbounds CuDynamicSharedArray(::Type{T}, dims) where {T} =
    CuDynamicSharedArray(T, dims, 0)

macro cuDynamicSharedMem(T, dims, offset=0)
    Base.depwarn("@cuDynamicSharedMem is deprecated, please use the CuDynamicSharedArray function", :CuStaticSharedArray)
    quote
        CuDynamicSharedArray($(esc(T)), $(esc(dims)), $(esc(offset)))
    end
end

dynamic_smem_size() =
    @asmcall("mov.u32 \$0, %dynamic_smem_size;", "=r", true, UInt32, Tuple{})

# get a pointer to shared memory, with known (static) or zero length (dynamic shared memory)
@generated function emit_shmem(::Type{T}, ::Val{len}=Val(0)) where {T,len}
    @dispose ctx=Context() begin
        T_int8 = LLVM.Int8Type()
        T_ptr = convert(LLVMType, LLVMPtr{T,AS.Shared})

        # create a function
        llvm_f, _ = create_function(T_ptr)

        # determine the array size
        # TODO: assert that allocatedinline(T) (or it won't have a layout)
        sz = len*sizeof(T)
        if !isbitstype(T)
            sz += len
        end

        # create the global variable
        # NOTE: this variable can't have T as element type, because it may be a boxed type
        #       when we're dealing with a union isbits array (e.g. `Union{Missing,Int}`)
        mod = LLVM.parent(llvm_f)
        gv_typ = LLVM.ArrayType(T_int8, sz)
        gv = GlobalVariable(mod, gv_typ, "shmem", AS.Shared)
        if len > 0
            # static shared memory should be demoted to local variables, whenever possible.
            # this is done by the NVPTX ASM printer:
            # > Find out if a global variable can be demoted to local scope.
            # > Currently, this is valid for CUDA shared variables, which have local
            # > scope and global lifetime. So the conditions to check are :
            # > 1. Is the global variable in shared address space?
            # > 2. Does it have internal linkage?
            # > 3. Is the global variable referenced only in one function?
            linkage!(gv, LLVM.API.LLVMInternalLinkage)
            initializer!(gv, null(gv_typ))
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
        alignment!(gv, align)

        # generate IR
        @dispose builder=IRBuilder() begin
            entry = BasicBlock(llvm_f, "entry")
            position!(builder, entry)

            ptr = gep!(builder, gv_typ, gv, [ConstantInt(0), ConstantInt(0)])

            untyped_ptr = bitcast!(builder, ptr, T_ptr)

            ret!(builder, untyped_ptr)
        end

        call_function(llvm_f, LLVMPtr{T,AS.Shared})
    end
end


# Dynamic Global Memory Allocation and Operations (B.21)

export malloc

@generated function malloc(sz::Csize_t)
    @dispose ctx=Context() begin
        T_pint8 = LLVM.PointerType(LLVM.Int8Type())
        T_size = convert(LLVMType, Csize_t)
        T_ptr = convert(LLVMType, Ptr{Cvoid})

        # create function
        llvm_f, _ = create_function(T_ptr, [T_size])
        mod = LLVM.parent(llvm_f)

        # get the intrinsic
        # NOTE: LLVM doesn't have void*, Clang uses i8* for malloc too
        intr_typ = LLVM.FunctionType(T_pint8, [T_size])
        intr = LLVM.Function(mod, "malloc", intr_typ)
        # should we attach some metadata here? julia.gc_alloc_obj has the following:
        #let attrs = function_attributes(intr)
        #    AllocSizeNumElemsNotPresent = reinterpret(Cuint, Cint(-1))
        #    packed_allocsize = Int64(1) << 32 | AllocSizeNumElemsNotPresent
        #    push!(attrs, EnumAttribute("allocsize", packed_allocsize))
        #end
        #let attrs = return_attributes(intr)
        #    push!(attrs, EnumAttribute("noalias", 0))
        #    push!(attrs, EnumAttribute("nonnull", 0))
        #end

        # generate IR
        @dispose builder=IRBuilder() begin
            entry = BasicBlock(llvm_f, "entry")
            position!(builder, entry)

            ptr = call!(builder, intr_typ, intr, [parameters(llvm_f)[1]])

            jlptr = ptrtoint!(builder, ptr, T_ptr)

            ret!(builder, jlptr)
        end

        call_function(llvm_f, Ptr{Cvoid}, Tuple{Csize_t}, :sz)
    end
end
