# Constant Memory

export CuDeviceConstantMemory

"""
    CuDeviceConstantMemory{T,N,Name,Shape}

The device-side counterpart of [`CuConstantMemory{T,N}`](@ref). This type should not be used
directly except in the case of `CuConstantMemory` global variables, where it denotes the
type of the relevant kernel argument.

Note that the `Name` and `Shape` type variables are implementation details and it
discouraged to use them directly. Instead use [name(::CuConstantMemory)](@ref) and
[Base.size(::CuConstantMemory)](@ref) respectively.
"""
struct CuDeviceConstantMemory{T,N,Name,Shape} <: AbstractArray{T,N} end

"""
Get the name of underlying global variable of this `CuDeviceConstantMemory`.
"""
name(::CuDeviceConstantMemory{T,N,Name,Shape}) where {T,N,Name,Shape} = Name

Base.:(==)(A::CuDeviceConstantMemory, B::CuDeviceConstantMemory) = name(A) == name(B)
Base.hash(A::CuDeviceConstantMemory, h::UInt) = hash(name(A), h)

Base.size(::CuDeviceConstantMemory{T,N,Name,Shape}) where {T,N,Name,Shape} = Shape

Base.@propagate_inbounds Base.getindex(A::CuDeviceConstantMemory, i::Integer) = constmemref(A, i)

Base.IndexStyle(::Type{<:CuDeviceConstantMemory}) = Base.IndexLinear()

@inline function constmemref(A::CuDeviceConstantMemory{T,N,Name,Shape}, index::Integer) where {T,N,Name,Shape}
    @boundscheck checkbounds(A, index)
    len = length(A)
    return read_constant_mem(Val(Name), index, T, Val(len))
end

@generated function read_constant_mem(::Val{global_name}, index::Integer, ::Type{T}, ::Val{len}) where {global_name,T,len}
    JuliaContext() do ctx
        # define LLVM types
        T_int = convert(LLVMType, Int, ctx)
        T_result = convert(LLVMType, T, ctx)

        # define function and get LLVM module
        param_types = [T_int]
        llvm_f, _ = create_function(T_result, param_types)
        mod = LLVM.parent(llvm_f)

        # create a constant memory global variable
        T_global = LLVM.ArrayType(T_result, len)
        global_var = GlobalVariable(mod, T_global, string(global_name), AS.Constant)
        linkage!(global_var, LLVM.API.LLVMExternalLinkage) # NOTE: external linkage is the default
        extinit!(global_var, true)
        # TODO: global_var alignment?

        # generate IR
        Builder(ctx) do builder
            entry = BasicBlock(llvm_f, "entry", ctx)
            position!(builder, entry)

            typed_ptr = inbounds_gep!(builder, global_var, [ConstantInt(0, ctx), parameters(llvm_f)[1]])
            ld = load!(builder, typed_ptr)

            metadata(ld)[LLVM.MD_tbaa] = tbaa_addrspace(AS.Constant, ctx)

            ret!(builder, ld)
        end

        # call the function
        call_function(llvm_f, T, Tuple{Int}, :((Int(index - one(index))),))
    end
end
