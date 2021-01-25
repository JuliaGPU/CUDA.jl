export CuConstantMemory

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
mutable struct CuConstantMemory{T,N} <: AbstractArray{T,N}
    name::String
    size::Dims{N}
    value::Union{Nothing,Array{T,N}}

    function CuConstantMemory(value::Array{T,N}; name::String) where {T,N}
        Base.isbitstype(T) || throw(ArgumentError("CuConstantMemory only supports bits types"))
        return new{T,N}(GPUCompiler.safe_name("constant_" * name), size(value), deepcopy(value))
    end

    function CuConstantMemory(::UndefInitializer, dims::Dims{N}; name::String) where {T,N}
        Base.isbitstype(T) || throw(ArgumentError("CuConstantMemory only supports bits types"))
        return new{T,N}(GPUCompiler.safe_name("constant_" * name), dims, nothing)
    end
end

CuConstantMemory{T}(::UndefInitializer, dims::Integer...; kwargs...) where {T} =
    CuConstantMemory(Array{T}(undef, dims); kwargs...)
CuConstantMemory{T}(::UndefInitializer, dims::Dims{N}; kwargs...) where {T,N} =
    CuConstantMemory{T,N}(undef, dims; kwargs...)

Base.size(A::CuConstantMemory) = A.size

Base.getindex(A::CuConstantMemory, i::Integer) = Base.getindex(A.value, i)
Base.setindex!(A::CuConstantMemory, v, i::Integer) = Base.setindex!(A.value, v, i)
Base.IndexStyle(::Type{<:CuConstantMemory}) = Base.IndexLinear()

function Adapt.adapt_storage(::Adaptor, A::CuConstantMemory{T,N}) where {T,N}
    # convert the values to the type domain
    # XXX: this is tough on the compiler when dealing with large initializers.
    typevals = if A.value !== nothing
        Tuple(reshape(A.value, prod(A.size)))
    else
        nothing
    end

    CuDeviceConstantMemory{T,N,Symbol(A.name),A.size,typevals}()
end


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
