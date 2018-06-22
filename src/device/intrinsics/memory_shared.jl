# Shared Memory (part of B.2)

export @cuStaticSharedMem, @cuDynamicSharedMem

# FIXME: `shmem_id` increment in the macro isn't correct, as multiple parametrically typed
#        functions will alias the id (but the size might be a parameter). but incrementing in
#        the @generated function doesn't work, as it is supposed to be pure and identical
#        invocations will erroneously share (and even cause multiple shmem globals).
shmem_id = 0

"""
    @cuStaticSharedMem(T::Type, dims) -> CuDeviceArray{T,AS.Shared}

Get an array of type `T` and dimensions `dims` (either an integer length or tuple shape)
pointing to a statically-allocated piece of shared memory. The type should be statically
inferable and the dimensions should be constant, or an error will be thrown and the
generator function will be called dynamically.
"""
macro cuStaticSharedMem(T, dims)
    global shmem_id
    id = shmem_id::Int += 1

    quote
        len = prod($(esc(dims)))
        ptr = _shmem(Val($id), $(esc(T)), Val(len))
        CuDeviceArray($(esc(dims)), DevicePtr{$(esc(T)), AS.Shared}(ptr))
    end
end

"""
    @cuDynamicSharedMem(T::Type, dims, offset::Integer=0) -> CuDeviceArray{T,AS.Shared}

Get an array of type `T` and dimensions `dims` (either an integer length or tuple shape)
pointing to a dynamically-allocated piece of shared memory. The type should be statically
inferable or an error will be thrown and the generator function will be called dynamically.

Note that the amount of dynamic shared memory needs to specified when launching the kernel.

Optionally, an offset parameter indicating how many bytes to add to the base shared memory
pointer can be specified. This is useful when dealing with a heterogeneous buffer of dynamic
shared memory; in the case of a homogeneous multi-part buffer it is preferred to use `view`.
"""
macro cuDynamicSharedMem(T, dims, offset=0)
    global shmem_id
    id = shmem_id::Int += 1

    # TODO: boundscheck against %dynamic_smem_size (currently unsupported by LLVM)

    quote
        len = prod($(esc(dims)))
        ptr = _shmem(Val($id), $(esc(T))) + $(esc(offset))
        CuDeviceArray($(esc(dims)), DevicePtr{$(esc(T)), AS.Shared}(ptr))
    end
end

# get a pointer to shared memory, with known (static) or zero length (dynamic shared memory)
@generated function _shmem(::Val{id}, ::Type{T}, ::Val{len}=Val(0)) where {id,T,len}
    eltyp = convert(LLVMType, T)

    T_ptr = convert(LLVMType, Ptr{T})
    T_actual_ptr = LLVM.PointerType(eltyp)

    # create a function
    llvm_f, _ = create_function(T_ptr)

    # create the global variable
    mod = LLVM.parent(llvm_f)
    gv_typ = LLVM.ArrayType(eltyp, len)
    gv = GlobalVariable(mod, gv_typ, "shmem$id", 3)
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
    # we pick 16 bytes since this is the largest transaction size as supported by PTX.
    alignment!(gv, max(16, datatype_align(T)))

    # generate IR
    Builder(JuliaContext()) do builder
        entry = BasicBlock(llvm_f, "entry", JuliaContext())
        position!(builder, entry)

        ptr_with_as = gep!(builder, gv, [ConstantInt(0, JuliaContext()),
                                         ConstantInt(0, JuliaContext())])

        ptr = addrspacecast!(builder, ptr_with_as, T_actual_ptr)
        val = ptrtoint!(builder, ptr, T_ptr)
        ret!(builder, val)
    end

    call_function(llvm_f, Ptr{T})
end
