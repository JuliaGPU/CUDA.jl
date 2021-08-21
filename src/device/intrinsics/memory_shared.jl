# Shared Memory (part of B.2)

export @cuStaticSharedMem, @cuDynamicSharedMem

shmem_id = 0

"""
    @cuStaticSharedMem(T::Type, dims) -> CuDeviceArray{T,AS.Shared}

Get an array of type `T` and dimensions `dims` (either an integer length or tuple shape)
pointing to a statically-allocated piece of shared memory. The type should be statically
inferable and the dimensions should be constant, or an error will be thrown and the
generator function will be called dynamically.
"""
macro cuStaticSharedMem(T, dims)
    # FIXME: generating a unique id in the macro is incorrect, as multiple parametrically typed
    #        functions will alias the id (and the size might be a parameter). but incrementing in
    #        the @generated function doesn't work, as it is supposed to be pure and identical
    #        invocations will erroneously share (and even cause multiple shmem globals).
    id = gensym("static_shmem")

    quote
        len = prod($(esc(dims)))
        ptr = emit_shmem(Val($(QuoteNode(id))), $(esc(T)), Val(len))
        CuDeviceArray($(esc(dims)), ptr)
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
    id = gensym("dynamic_shmem")

    # TODO: boundscheck against %dynamic_smem_size (currently unsupported by LLVM)

    quote
        len = prod($(esc(dims)))
        ptr = emit_shmem(Val($(QuoteNode(id))), $(esc(T))) + $(esc(offset))
        CuDeviceArray($(esc(dims)), ptr)
    end
end

# get a pointer to shared memory, with known (static) or zero length (dynamic shared memory)
@generated function emit_shmem(::Val{id}, ::Type{T}, ::Val{len}=Val(0)) where {id,T,len}
    Context() do ctx
        eltyp = convert(LLVMType, T; ctx)
        T_ptr = convert(LLVMType, LLVMPtr{T,AS.Shared}; ctx)

        # create a function
        llvm_f, _ = create_function(T_ptr)

        # create the global variable
        mod = LLVM.parent(llvm_f)
        gv_typ = LLVM.ArrayType(eltyp, len)
        gv = GlobalVariable(mod, gv_typ, GPUCompiler.safe_name(string(id)), AS.Shared)
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
        alignment!(gv, Base.max(32, Base.datatype_alignment(T)))

        # generate IR
        Builder(ctx) do builder
            entry = BasicBlock(llvm_f, "entry"; ctx)
            position!(builder, entry)

            ptr = gep!(builder, gv, [ConstantInt(0; ctx), ConstantInt(0; ctx)])

            untyped_ptr = bitcast!(builder, ptr, T_ptr)

            ret!(builder, untyped_ptr)
        end

        call_function(llvm_f, LLVMPtr{T,AS.Shared})
    end
end
