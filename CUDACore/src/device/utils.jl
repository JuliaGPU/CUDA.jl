# helpers for writing device functionality

# local method table for device functions
@static if isdefined(Base.Experimental, Symbol("@overlay"))
Base.Experimental.@MethodTable(method_table)
else
const method_table = nothing
end

@public @device_override, @device_function, @device_functions

macro device_override(ex)
    esc(GPUToolbox.Overlays.device_override(__module__, GlobalRef(CUDACore, :method_table), ex))
end

macro device_function(ex)
    esc(GPUToolbox.Overlays.device_function(__module__, GlobalRef(CUDACore, :method_table), ex))
end

# `device_function` also rewrites every definition in a block
macro device_functions(ex)
    esc(GPUToolbox.Overlays.device_function(__module__, GlobalRef(CUDACore, :method_table), ex))
end

## alignment API

# we don't expose this as Aligned{N}, because we want to have the T typevar first
# to facilitate use in function signatures as ::Aligned{<:T}

struct Aligned{T, N}
    data::T
end

alignment(::Aligned{<:Any, N}) where {N} = N
Base.getindex(x::Aligned) = x.data

"""
    CUDA.align{N}(obj)

Construct an aligned object, providing alignment information to APIs that require it.
"""
struct align{N} end
(::Type{align{N}})(data::T) where {T,N} = Aligned{T,N}(data)

# default alignment for common types
Aligned(x::Aligned) = x
Aligned(x::Ptr{T}) where T = align{Base.datatype_alignment(T)}(x)
Aligned(x::LLVMPtr{T}) where T = align{Base.datatype_alignment(T)}(x)
