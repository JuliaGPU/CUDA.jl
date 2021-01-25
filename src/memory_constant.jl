export CuConstantMemory

# Map a constant memory name to its array value
const constant_memory_initializer = Dict{Symbol,WeakRef}()

"""
    CuConstantMemory{T,N}(value::Array{T,N})
    CuConstantMemory{T}(::UndefInitializer, dims::Integer...)
    CuConstantMemory{T}(::UndefInitializer, dims::Dims{N})

Construct an `N`-dimensional constant memory array of type `T`, where `isbits(T)`.

Note that `deepcopy` will be called on the `value` constructor argument, meaning that
mutations to the original `value` or its elements after construction will not be
reflected in the value of `CuConstantMemory`.

The `UndefInitializer` constructors behave exactly like the regular `Array` version,
i.e. the value of `CuConstantMemory` will be completely random when using them.

Unlike in CUDA C, structs cannot be put directly into constant memory. This feature can
be emulated however by wrapping the struct inside of a 1-element array.

When using `CuConstantMemory` as a global variable it is required to pass it as an argument
to a kernel, where the argument is of type [`CuDeviceConstantMemory{T,N}`](@ref).
When using `CuConstantMemory` as a local variable that is captured by a kernel closure
this is not required, and it can be used directly like any other captured variable
without passing it as an argument.

In cases where the same kernel object gets called mutiple times, and it is desired to mutate
the value of a `CuConstantMemory` variable in this kernel between calls, please refer
to [`Base.copyto!(const_mem::CuConstantMemory{T}, value::Array{T}, kernel::HostKernel)`](@ref)
"""
struct CuConstantMemory{T,N} <: AbstractArray{T,N}
    name::Symbol
    value::Array{T,N}

    function CuConstantMemory(value::Array{T,N}) where {T,N}
        # TODO: add finalizer that removes the relevant entry from constant_memory_initializer?
        Base.isbitstype(T) || throw(ArgumentError("CuConstantMemory only supports bits types"))
        name = gensym("constant_memory")
        name = GPUCompiler.safe_name(string(name))
        name = Symbol(name)
        val = deepcopy(value)
        constant_memory_initializer[name] = WeakRef(val)
        return new{T,N}(name, val)
    end
end

CuConstantMemory{T}(::UndefInitializer, dims::Integer...) where {T} =
    CuConstantMemory(Array{T}(undef, dims))
CuConstantMemory{T}(::UndefInitializer, dims::Dims{N}) where {T,N} =
    CuConstantMemory(Array{T,N}(undef, dims))

Base.size(A::CuConstantMemory) = size(A.value)

Base.getindex(A::CuConstantMemory, i::Integer) = Base.getindex(A.value, i)
Base.setindex!(A::CuConstantMemory, v, i::Integer) = Base.setindex!(A.value, v, i)
Base.IndexStyle(::Type{<:CuConstantMemory}) = Base.IndexLinear()

Adapt.adapt_storage(::Adaptor, A::CuConstantMemory{T,N}) where {T,N} = 
    CuDeviceConstantMemory{T,N,A.name,size(A.value)}()


"""
Given a `kernel` returned by `@cuda`, copy `value` into `const_mem` for subsequent calls to this `kernel`.
If `const_mem` is not used within `kernel`, an error will be thrown.
"""
function Base.copyto!(const_mem::CuConstantMemory{T}, value::Array{T}, kernel::HostKernel) where T
    # TODO: add bool argument to also change the value field of const_mem?
    if size(const_mem) != size(value)
        throw(DimensionMismatch("size of `value` does not match size of constant memory"))
    end

    global_array = CuGlobalArray{T}(kernel.mod, string(const_mem.name), length(const_mem))
    copyto!(global_array, value)
end


function emit_constant_memory_initializer!(mod::LLVM.Module)
    for global_var in globals(mod)
        T_global = llvmtype(global_var)
        if addrspace(T_global) == AS.Constant
            constant_memory_name = Symbol(LLVM.name(global_var))
            if !haskey(constant_memory_initializer, constant_memory_name)
                continue # non user defined constant memory, most likely from the CUDA runtime
            end

            arr = constant_memory_initializer[constant_memory_name].value
            @assert !isnothing(arr) "calling kernel containing garbage collected constant memory"

            flattened_arr = reduce(vcat, arr)
            ctx = LLVM.context(mod)
            typ = eltype(eltype(T_global))

            # TODO: have a look at how julia converts structs to llvm:
            #       https://github.com/JuliaLang/julia/blob/80ace52b03d9476f3d3e6ff6da42f04a8df1cf7b/src/cgutils.cpp#L572
            #       this only seems to emit a type though
            if isa(typ, LLVM.IntegerType) || isa(typ, LLVM.FloatingPointType)
                init = ConstantArray(flattened_arr, ctx)
            elseif isa(typ, LLVM.ArrayType) # a struct with every field of the same type gets optimized to an array
                constant_arrays = LLVM.Constant[]
                for x in flattened_arr
                    fields = collect(map(name->getfield(x, name), fieldnames(typeof(x))))
                    constant_array = ConstantArray(fields, ctx)
                    push!(constant_arrays, constant_array)
                end
                init = ConstantArray(typ, constant_arrays)
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
                            throw(error("constant memory does not currently support structs with non-primitive fields ($(typeof(x)).$fieldname::$(typeof(field)))"))
                        end
                    end
                    const_struct = ConstantStruct(typ, constants)
                    push!(constant_structs, const_struct)
                end
                init = ConstantArray(typ, constant_structs)
            else
                # unreachable, but let's be safe and throw a nice error message just in case
                throw(error("could not emit initializer for constant memory of type $typ"))
            end
            
            initializer!(global_var, init)
        end
    end
end
