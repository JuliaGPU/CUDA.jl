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
struct CuDeviceConstantMemory{T,N,Name,Shape,Hash} <: AbstractArray{T,N} end

"""
Get the name of underlying global variable of this `CuDeviceConstantMemory`.
"""
name(::CuDeviceConstantMemory{T,N,Name}) where {T,N,Name} = Name

Base.:(==)(A::CuDeviceConstantMemory, B::CuDeviceConstantMemory) = name(A) == name(B)
Base.hash(A::CuDeviceConstantMemory, h::UInt) = hash(name(A), h)

Base.size(::CuDeviceConstantMemory{T,N,Name,Shape}) where {T,N,Name,Shape} = Shape

Base.@propagate_inbounds Base.getindex(A::CuDeviceConstantMemory, i::Integer) = constmemref(A, i)

Base.IndexStyle(::Type{<:CuDeviceConstantMemory}) = Base.IndexLinear()

@inline function constmemref(A::CuDeviceConstantMemory{T,N,Name,Shape,Init}, index::Integer) where {T,N,Name,Shape,Init}
    @boundscheck checkbounds(A, index)
    len = length(A)
    return read_constant_mem(Val(Name), index, T, Val(Shape), Val(Init))
end

@generated function read_constant_mem(::Val{global_name}, index::Integer, ::Type{T}, ::Val{shape}, ::Val{init}) where {global_name,T,shape,init}
    JuliaContext() do ctx
        # define LLVM types
        T_int = convert(LLVMType, Int, ctx)
        T_result = convert(LLVMType, T, ctx)

        # define function and get LLVM module
        param_types = [T_int]
        llvm_f, _ = create_function(T_result, param_types)
        mod = LLVM.parent(llvm_f)

        # create a constant memory global variable
        # TODO: global_var alignment?
        len = prod(shape)
        T_global = LLVM.ArrayType(T_result, len)
        global_var = GlobalVariable(mod, T_global, string(global_name), AS.Constant)
        linkage!(global_var, LLVM.API.LLVMWeakAnyLinkage) # merge, but make sure symbols aren't discarded
        extinit!(global_var, true)
        # XXX: if we don't extinit, LLVM can inline the constant memory if it's predefined.
        #      that means we wouldn't be able to re-set it afterwards. do we want that?

        # initialize the constant memory
        if init !== nothing
            arr = reshape([init...], shape)
            if isnothing(arr)
                GPUCompiler.@safe_error "calling kernel containing garbage collected constant memory"
            end

            flattened_arr = reduce(vcat, arr)
            typ = eltype(T_global)

            # TODO: have a look at how julia converts structs to llvm:
            #       https://github.com/JuliaLang/julia/blob/80ace52b03d9476f3d3e6ff6da42f04a8df1cf7b/src/cgutils.cpp#L572
            #       this only seems to emit a type though
            init = if isa(typ, LLVM.IntegerType) || isa(typ, LLVM.FloatingPointType)
                ConstantArray(flattened_arr, ctx)
            elseif isa(typ, LLVM.ArrayType) # a struct with every field of the same type gets optimized to an array
                constant_arrays = LLVM.Constant[]
                for x in flattened_arr
                    fields = collect(map(name->getfield(x, name), fieldnames(typeof(x))))
                    constant_array = ConstantArray(fields, ctx)
                    push!(constant_arrays, constant_array)
                end
                ConstantArray(typ, constant_arrays)
            elseif isa(typ, LLVM.StructType)
                constant_structs = LLVM.Constant[]
                for x in flattened_arr
                    constants = LLVM.Constant[]
                    for fieldname in fieldnames(typeof(x))
                        field = getfield(x, fieldname)
                        if isa(field, Bool)
                            # NOTE: Bools get compiled to i8 instead of the more "correct" type i1
                            push!(constants, ConstantInt(LLVM.Int8Type(ctx), field))
                        elseif isa(field, Integer)
                            push!(constants, ConstantInt(field, ctx))
                        elseif isa(field, AbstractFloat)
                            push!(constants, ConstantFP(field, ctx))
                        else
                            GPUCompiler.@safe_error "constant memory does not currently support structs with non-primitive fields ($(typeof(x)).$fieldname::$(typeof(field)))"
                        end
                    end
                    const_struct = ConstantStruct(typ, constants)
                    push!(constant_structs, const_struct)
                end
                ConstantArray(typ, constant_structs)
            else
                # unreachable, but let's be safe and throw a nice error message just in case
                GPUCompiler.@safe_error "Could not emit initializer for constant memory of type $typ"
                nothing
            end

            init !== nothing && initializer!(global_var, init)
        end

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
