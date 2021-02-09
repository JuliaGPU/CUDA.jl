export CuConstantMemory

"""
Generate a name based on lexical information
"""
macro generate_name()
    quote
        sts = StackTraces.lookup.(backtrace())
        filter!(st -> !first(st).from_c, sts)
        parent_frame = first(sts[2])
        func = string(parent_frame.func)
        file = splitpath(string(parent_frame.file))[end]
        line = string(parent_frame.line)
        GPUCompiler.safe_name("constant_memory_" * func * "_" * file * "_" * line)
    end
end

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
    value::Array{T,N}
    track_value_between_kernels::Bool

    function CuConstantMemory(value::Array{T,N}; name::String=@generate_name(), track_value_between_kernels::Bool=true) where {T,N}
        Base.isbitstype(T) || throw(ArgumentError("CuConstantMemory only supports bits types"))
        return new{T,N}(name, size(value), deepcopy(value), track_value_between_kernels)
    end
end

CuConstantMemory{T}(::UndefInitializer, dims::Integer...; name::String=@generate_name(), kwargs...) where {T} =
    CuConstantMemory(Array{T}(undef, dims); name=name, kwargs...)
CuConstantMemory{T}(::UndefInitializer, dims::Dims{N}; name::String=@generate_name(), kwargs...) where {T,N} =
    CuConstantMemory(Array{T,N}(undef, dims); name=name, kwargs...)

Base.size(A::CuConstantMemory) = A.size

Base.getindex(A::CuConstantMemory, i::Integer) = Base.getindex(A.value, i)
Base.setindex!(A::CuConstantMemory, v, i::Integer) = Base.setindex!(A.value, v, i)
Base.IndexStyle(::Type{<:CuConstantMemory}) = Base.IndexLinear()

Adapt.adapt_storage(::Adaptor, A::CuConstantMemory{T,N}) where {T,N} = 
    CuDeviceConstantMemory{T,N,Symbol(A.name),A.size}()

"""
Given a `kernel` returned by `@cuda`, copy `value` into `const_mem` for subsequent calls to this `kernel`.
If `propagate_to_host` is `true`, also change the `value` field of `const_mem`.
If `const_mem` is not used within `kernel`, an error will be thrown.
"""
function Base.copyto!(const_mem::CuConstantMemory{T}, value::Array{T}, kernel::HostKernel, propagate_to_host::Bool=true) where T
    if size(const_mem) != size(value)
        throw(DimensionMismatch("size of `value` does not match size of constant memory"))
    end

    if propagate_to_host
        const_mem.value = deepcopy(value)
    end

    global_array = CuGlobalArray{T}(kernel.mod, string(const_mem.name), length(const_mem))
    copyto!(global_array, value)
end
