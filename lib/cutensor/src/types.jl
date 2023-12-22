## data types

const scalar_types = Dict(
    (Float16, Float16)          => Float32,
    (Float32, Float16)          => Float32,
    (Float16, Float32)          => Float32,
    (Float32, Float32)          => Float32,
    (Float64, Float64)          => Float64,
    (Float64, Float32)          => Float64,
    (ComplexF32, ComplexF32)    => ComplexF32,
    (ComplexF64, ComplexF64)    => ComplexF64,
    (ComplexF64, ComplexF32)    => ComplexF64)

function Base.cconvert(::Type{cutensorComputeDescriptor_t}, T::DataType)
    if T == Float16 || T == ComplexF16
        return CUTENSOR_COMPUTE_DESC_16F()
    elseif T == Float32 || T == ComplexF32
        return CUTENSOR_COMPUTE_DESC_32F()
    elseif T == Float64 || T == ComplexF64
        return CUTENSOR_COMPUTE_DESC_64F()
    else
        throw(ArgumentError("cutensorComputeType equivalent for input type $T does not exist!"))
    end
end

function Base.convert(::Type{cutensorDataType_t}, T::DataType)
    if T == Float16
        return CUTENSOR_R_16F
    elseif T == ComplexF16
        return CUTENSOR_C_16F
    elseif T == Float32
        return CUTENSOR_R_32F
    elseif T == ComplexF32
        return CUTENSOR_C_32F
    elseif T == Float64
        return CUTENSOR_R_64F
    elseif T == ComplexF64
        return CUTENSOR_C_64F
    elseif T == Int8
        return CUTENSOR_R_8I
    elseif T == Int32
        return CUTENSOR_R_32I
    elseif T == UInt8
        return CUTENSOR_R_8U
    elseif T == UInt32
        return CUTENSOR_R_32U
    else
        throw(ArgumentError("cutensorDataType equivalent for input type $T does not exist!"))
    end
end


## plan

mutable struct CuTensorPlan
    ctx::CuContext
    handle::cutensorPlan_t
    workspace::CuVector{UInt8,Mem.DeviceBuffer}

    function CuTensorPlan(desc, pref; workspacePref=CUTENSOR_WORKSPACE_DEFAULT)
        # estimate the workspace size
        workspaceSizeEstimate = Ref{UInt64}()
        cutensorEstimateWorkspaceSize(handle(), desc, pref, workspacePref, workspaceSizeEstimate)

        # create the plan
        plan_ref = Ref{cutensorPlan_t}()
        cutensorCreatePlan(handle(), plan_ref, desc, pref, workspaceSizeEstimate[])

        # allocate the actual workspace
        actualWorkspaceSize = Ref{UInt64}()
        cutensorPlanGetAttribute(handle(), plan_ref[], CUTENSOR_PLAN_REQUIRED_WORKSPACE, actualWorkspaceSize, sizeof(actualWorkspaceSize))
        workspace = CuArray{UInt8}(undef, actualWorkspaceSize[])

        obj = new(context(), plan_ref[], workspace)
        finalizer(CUDA.unsafe_finalize!, obj)
        return obj
    end
end

Base.unsafe_convert(::Type{cutensorPlan_t}, plan::CuTensorPlan) = plan.handle

Base.:(==)(a::CuTensorPlan, b::CuTensorPlan) = a.handle == b.handle
Base.hash(plan::CuTensorPlan, h::UInt) = hash(plan.handle, h)

# destroying the plan
function unsafe_destroy!(plan::CuTensorPlan)
    context!(plan.ctx; skip_destroyed=true) do
        cutensorDestroyPlan(plan)
    end
end

# early freeing the plan and associated workspace
function CUDA.unsafe_free!(plan::CuTensorPlan)
    CUDA.unsafe_free!(plan.workspace)
    if plan.handle != C_NULL
        unsafe_destroy!(plan)
        plan.handle = C_NULL
    end
end

# GC-driven freeing of the plan and associated workspace
function CUDA.unsafe_finalize!(plan::CuTensorPlan)
    CUDA.unsafe_finalize!(plan.workspace)
    if plan.handle != C_NULL
        unsafe_destroy!(plan)
        plan.handle = C_NULL
    end
end


## descriptor

mutable struct CuTensorDescriptor
    handle::cutensorTensorDescriptor_t

    function CuTensorDescriptor(a; size = size(a), strides = strides(a), eltype = eltype(a))
        sz = collect(Int64, size)
        st = collect(Int64, strides)
        alignmentRequirement::UInt32 = 128

        desc = Ref{cutensorTensorDescriptor_t}()
        cutensorCreateTensorDescriptor(handle(), desc, length(sz), sz, st, eltype, alignmentRequirement)

        obj = new(desc[])
        finalizer(unsafe_destroy!, obj)
        return obj
    end
end

Base.unsafe_convert(::Type{cutensorTensorDescriptor_t}, obj::CuTensorDescriptor) = obj.handle

function unsafe_destroy!(obj::CuTensorDescriptor)
    cutensorDestroyTensorDescriptor(obj)
end


## tensor

export CuTensor

mutable struct CuTensor{T, N}
    data::DenseCuArray{T, N}
    inds::Vector{Char}
    function CuTensor{T, N}(data::DenseCuArray{T, N}, inds::Vector{Char}) where {T<:Number, N}
        new(data, inds)
    end
    function CuTensor{T, N}(data::DenseCuArray{N, T}, inds::Vector{<:AbstractChar}) where {T<:Number, N}
        new(data, Char.(inds))
    end
end

CuTensor(data::DenseCuArray{T, N}, inds::Vector{<:AbstractChar}) where {T<:Number, N} =
    CuTensor{T, N}(data, convert(Vector{Char}, inds))

CuTensor(data::DenseCuArray{T, N}, inds::Vector{Char}) where {T<:Number, N} =
    CuTensor{T, N}(data, inds)

# array interface
Base.size(T::CuTensor) = size(T.data)
Base.size(T::CuTensor, i) = size(T.data, i)
Base.length(T::CuTensor) = length(T.data)
Base.ndims(T::CuTensor) = length(T.inds)
Base.strides(T::CuTensor) = strides(T.data)
Base.eltype(T::CuTensor) = eltype(T.data)
Base.similar(T::CuTensor{Tv, N}) where {Tv, N} = CuTensor{Tv, N}(similar(T.data), copy(T.inds))
Base.copy(T::CuTensor{Tv, N}) where {Tv, N} = CuTensor{Tv, N}(copy(T.data), copy(T.inds))
Base.collect(T::CuTensor) = (collect(T.data), T.inds)
