export CuGlobalMemory

macro generate_name_global()
    quote
        sts = StackTraces.lookup.(backtrace())
        filter!(st -> !first(st).from_c, sts)
        parent_frame = first(sts[2])
        func = string(parent_frame.func)
        file = splitpath(string(parent_frame.file))[end]
        line = string(parent_frame.line)
        GPUCompiler.safe_name("global_memory_" * func * "_" * file * "_" * line)
    end
end


mutable struct CuGlobalMemory{T,N} <: AbstractArray{T,N}
    name::String
    size::Dims{N}
    value::Array{T,N}
    track_value_between_kernels::Bool

    function CuGlobalMemory(value::Array{T,N}; name::String=@generate_name_global(), track_value_between_kernels::Bool=true) where {T,N}
        Base.isbitstype(T) || throw(ArgumentError("CuGlobalMemory only supports bits types"))
        return new{T,N}(name, size(value), deepcopy(value), track_value_between_kernels)
    end
end

CuGlobalMemory{T}(::UndefInitializer, dims::Integer...; name::String=@generate_name_global(), kwargs...) where {T} =
    CuGlobalMemory(Array{T}(undef, dims); name=name, kwargs...)
CuGlobalMemory{T}(::UndefInitializer, dims::Dims{N}; name::String=@generate_name_global(), kwargs...) where {T,N} =
    CuGlobalMemory(Array{T,N}(undef, dims); name=name, kwargs...)

Base.size(A::CuGlobalMemory) = A.size

Base.getindex(A::CuGlobalMemory, i::Integer) = Base.getindex(A.value, i)
Base.setindex!(A::CuGlobalMemory, v, i::Integer) = Base.setindex!(A.value, v, i)
Base.IndexStyle(::Type{<:CuGlobalMemory}) = Base.IndexLinear()

Adapt.adapt_storage(::Adaptor, A::CuGlobalMemory{T,N}) where {T,N} = 
    CuDeviceGlobalMemory{T,N,Symbol(A.name),A.size}()


function Base.copyto!(global_mem::CuGlobalMemory{T}, value::Array{T}, kernel::HostKernel, propagate_to_host::Bool=true) where T
    if size(global_mem) != size(value)
        throw(DimensionMismatch("size of `value` does not match size of global memory"))
    end

    if propagate_to_host
        global_mem.value = deepcopy(value)
    end

    global_array = CuGlobalArray{T}(kernel.mod, string(global_mem.name), length(global_mem))
    copyto!(global_array, value)
end

