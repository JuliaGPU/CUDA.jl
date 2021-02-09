# Statically Allocated Global Memory

export CuDeviceGlobalMemory

struct CuDeviceGlobalMemory{T,N,Name,Shape} <: AbstractArray{T,N} end

name(::CuDeviceGlobalMemory{T,N,Name,Shape}) where {T,N,Name,Shape} = Name

Base.:(==)(A::CuDeviceGlobalMemory, B::CuDeviceGlobalMemory) = name(A) == name(B)
Base.hash(A::CuDeviceGlobalMemory, h::UInt) = hash(name(A), h)

Base.size(::CuDeviceGlobalMemory{T,N,Name,Shape}) where {T,N,Name,Shape} = Shape

Base.@propagate_inbounds Base.getindex(A::CuDeviceGlobalMemory, i::Integer) = 
    globalmemref(A, i)
Base.@propagate_inbounds Base.setindex!(A::CuDeviceGlobalMemory{T}, x, i::Integer) where {T} =
    globalmemset(A, convert(T, x), i)

Base.IndexStyle(::Type{<:CuDeviceGlobalMemory}) = Base.IndexLinear()

@inline function globalmemref(A::CuDeviceGlobalMemory{T,N,Name,Shape}, index::Integer) where {T,N,Name,Shape}
    @boundscheck checkbounds(A, index)
    len = length(A)
    return read_global_mem(Val(Name), index, T, Val(len))
end

@inline function globalmemset(A::CuDeviceGlobalMemory{T,N,Name,Shape}, x::T, index::Integer) where {T,N,Name,Shape}
    @boundscheck checkbounds(A, index)
    len = length(A)
    write_global_mem(Val(Name), index, x, Val(len))
    return A
end

@generated function read_global_mem(::Val{global_name}, index::Integer, ::Type{T}, ::Val{len}) where {global_name,T,len}
    JuliaContext() do ctx
        # define LLVM types
        T_int = convert(LLVMType, Int, ctx)
        T_result = convert(LLVMType, T, ctx)

        # define function and get LLVM module
        param_types = [T_int]
        llvm_f, _ = create_function(T_result, param_types)
        mod = LLVM.parent(llvm_f)

        # create a global memory global variable
        # TODO: global_var alignment?
        T_global = LLVM.ArrayType(T_result, len)
        global_var = GlobalVariable(mod, T_global, string(global_name), AS.Global)
        linkage!(global_var, LLVM.API.LLVMWeakAnyLinkage) # merge, but make sure symbols aren't discarded
        initializer!(global_var, null(T_global))
        extinit!(global_var, true)
        # XXX: if we don't extinit, LLVM can inline the constant memory if it's predefined.
        #      that means we wouldn't be able to re-set it afterwards. do we want that?

        # generate IR
        Builder(ctx) do builder
            entry = BasicBlock(llvm_f, "entry", ctx)
            position!(builder, entry)

            typed_ptr = inbounds_gep!(builder, global_var, [ConstantInt(0, ctx), parameters(llvm_f)[1]])
            ld = load!(builder, typed_ptr)

            metadata(ld)[LLVM.MD_tbaa] = tbaa_addrspace(AS.Global, ctx)

            ret!(builder, ld)
        end

        # call the function
        call_function(llvm_f, T, Tuple{Int}, :((Int(index - one(index))),))
    end
end

@generated function write_global_mem(::Val{global_name}, index::Integer, x::T, ::Val{len}) where {global_name,T,len}
    JuliaContext() do ctx
        # define LLVM types
        T_int = convert(LLVMType, Int, ctx)
        eltyp = convert(LLVMType, T, ctx)

        # define function and get LLVM module
        param_types = [eltyp, T_int]
        llvm_f, _ = create_function(LLVM.VoidType(ctx), param_types)
        mod = LLVM.parent(llvm_f)

        # create a global memory global variable
        # TODO: global_var alignment?
        T_global = LLVM.ArrayType(eltyp, len)
        global_var = GlobalVariable(mod, T_global, string(global_name), AS.Global)
        linkage!(global_var, LLVM.API.LLVMWeakAnyLinkage) # merge, but make sure symbols aren't discarded
        initializer!(global_var, null(T_global))
        extinit!(global_var, true)

        # generate IR
        Builder(ctx) do builder
            entry = BasicBlock(llvm_f, "entry", ctx)
            position!(builder, entry)

            typed_ptr = inbounds_gep!(builder, global_var, [ConstantInt(0, ctx), parameters(llvm_f)[2]])
            val = parameters(llvm_f)[1]
            st = store!(builder, val, typed_ptr)

            metadata(st)[LLVM.MD_tbaa] = tbaa_addrspace(AS.Global, ctx)

            ret!(builder)
        end

        # call the function
        call_function(llvm_f, Cvoid, Tuple{T, Int}, :((x, Int(index - one(index)))))
    end
end
