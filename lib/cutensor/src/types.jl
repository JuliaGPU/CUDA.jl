## data types

@enum cutensorComputeDescriptorEnum begin
    COMPUTE_DESC_16F = 1
    COMPUTE_DESC_32F = 2
    COMPUTE_DESC_TF32 = 3
    COMPUTE_DESC_3XTF32 = 4
    COMPUTE_DESC_64F = 5
end

const contraction_compute_types = Dict(
    # typeA,     typeB,      typeC       => typeCompute
    (Float16,    Float16,    Float16)    => Float32,
    (Float32,    Float32,    Float32)    => Float32,
    (Float64,    Float64,    Float64)    => Float64,
    (ComplexF32, ComplexF32, ComplexF32) => Float32,
    (ComplexF64, ComplexF64, ComplexF64) => Float64,
    (Float64,    ComplexF64, ComplexF64) => Float64,
    (ComplexF64, Float64,    ComplexF64) => Float64)

const elementwise_trinary_compute_types = Dict(
    # typeA,     typeB,      typeC       => typeCompute
    (Float16,    Float16,    Float16)    => Float16,
    (Float32,    Float32,    Float32)    => Float32,
    (Float64,    Float64,    Float64)    => Float64,
    (ComplexF32, ComplexF32, ComplexF32) => Float32,
    (ComplexF64, ComplexF64, ComplexF64) => Float64,
    (Float32,    Float32,    Float16)    => Float32,
  # (Float64,    Float64,    Float32)    => Float32,
    (ComplexF64, ComplexF64, ComplexF32) => Float64)

const elementwise_binary_compute_types = Dict(
    # typeA,     typeC       => typeCompute
    (Float16,    Float16)    => Float16,
    (Float32,    Float32)    => Float32,
    (Float64,    Float64)    => Float64,
    (ComplexF32, ComplexF32) => Float32,
    (ComplexF64, ComplexF64) => Float64,
    (ComplexF64, ComplexF32) => Float64,
    (Float32,    Float16)    => Float32,
    (Float64,    Float32)    => Float64)

const permutation_compute_types = Dict(
    # typeA,     typeB       => typeCompute
    (Float16,    Float16)    => Float16,
    (Float16,    Float32)    => Float32,
  # (Float32,    Float16)    => Float32,
    (Float32,    Float32)    => Float32,
    (Float64,    Float64)    => Float64,
    (Float32,    Float64)    => Float64,
  # (Float64,    Float32)    => Float64,
    (ComplexF32, ComplexF32) => Float32,
    (ComplexF64, ComplexF64) => Float64,
    (ComplexF32, ComplexF64) => Float64,
  # (ComplexF64, ComplexF32) => Float64
    )

const reduction_compute_types = Dict(
    # typeA,     typeC       => typeCompute
    (Float16,    Float16)    => Float16,
    (Float32,    Float32)    => Float32,
    (Float64,    Float64)    => Float64,
    (ComplexF32, ComplexF32) => Float32,
    (ComplexF64, ComplexF64) => Float64)

# we have our own enum to represent cutensorComputeDescriptor_t values
function Base.convert(::Type{cutensorComputeDescriptorEnum}, T::DataType)
    if T == Float16
        return COMPUTE_DESC_16F
    elseif T == Float32 || T == ComplexF32
        return COMPUTE_DESC_32F
    elseif T == Float64 || T == ComplexF64
        return COMPUTE_DESC_64F
    else
        throw(ArgumentError("cutensorComputeDescriptor equivalent for input type $T does not exist!"))
    end
end
Base.cconvert(::Type{cutensorComputeDescriptor_t}, T::DataType) =
    Base.cconvert(cutensorComputeDescriptor_t, convert(cutensorComputeDescriptorEnum, T))

function Base.cconvert(::Type{cutensorComputeDescriptor_t}, T::cutensorComputeDescriptorEnum)
    if T == COMPUTE_DESC_16F
        return CUTENSOR_COMPUTE_DESC_16F()
    elseif T == COMPUTE_DESC_32F
        return CUTENSOR_COMPUTE_DESC_32F()
    elseif T == COMPUTE_DESC_TF32
        return CUTENSOR_COMPUTE_DESC_TF32()
    elseif T == COMPUTE_DESC_3XTF32
        return CUTENSOR_COMPUTE_DESC_3XTF32()
    elseif T == COMPUTE_DESC_64F
        return CUTENSOR_COMPUTE_DESC_64F()
    else
        throw(ArgumentError("cutensorComputeDescriptor equivalent for input enum value $T does not exist!"))
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

function Base.convert(::DataType, T::cutensorDataType_t)
    if T == CUTENSOR_R_16F
        return Float16
    elseif T == CUTENSOR_R_32F
        return Float32
    elseif T == CUTENSOR_C_32F
        return ComplexF32
    elseif T == CUTENSOR_R_64F
        return Float64
    elseif T == CUTENSOR_C_64F
        return ComplexF64
    else
        throw(ArgumentError("Data type equivalent for cutensorDataType type $T does not exist!"))
    end
end


## plan

mutable struct CuTensorPlan
    ctx::CuContext
    handle::cutensorPlan_t
    workspace::CuVector{UInt8,CUDA.DeviceMemory}
    scalar_type::DataType

    function CuTensorPlan(desc, pref; workspacePref=CUTENSOR_WORKSPACE_DEFAULT)
        # estimate the workspace size
        workspaceSizeEstimate = Ref{UInt64}()
        cutensorEstimateWorkspaceSize(handle(), desc, pref, workspacePref, workspaceSizeEstimate)

        # determine the scalar type
        required_scalar_type = Ref{cutensorDataType_t}()
        cutensorOperationDescriptorGetAttribute(handle(), desc, CUTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE, required_scalar_type, sizeof(required_scalar_type))

        # create the plan
        plan_ref = Ref{cutensorPlan_t}()
        cutensorCreatePlan(handle(), plan_ref, desc, pref, workspaceSizeEstimate[])

        # allocate the actual workspace
        actualWorkspaceSize = Ref{UInt64}()
        cutensorPlanGetAttribute(handle(), plan_ref[], CUTENSOR_PLAN_REQUIRED_WORKSPACE, actualWorkspaceSize, sizeof(actualWorkspaceSize))
        workspace = CuArray{UInt8}(undef, actualWorkspaceSize[])

        obj = new(context(), plan_ref[], workspace, required_scalar_type[])
        finalizer(CUDA.unsafe_free!, obj)
        return obj
    end
end

Base.show(io::IO, plan::CuTensorPlan) = @printf(io, "CuTensorPlan(%p)", plan.handle)

Base.unsafe_convert(::Type{cutensorPlan_t}, plan::CuTensorPlan) = plan.handle

Base.:(==)(a::CuTensorPlan, b::CuTensorPlan) = a.handle == b.handle
Base.hash(plan::CuTensorPlan, h::UInt) = hash(plan.handle, h)

# destroying the plan
function unsafe_destroy!(plan::CuTensorPlan)
    context!(plan.ctx; skip_destroyed=true) do
        cutensorDestroyPlan(plan)
    end
end

# freeing the plan and associated workspace
function CUDA.unsafe_free!(plan::CuTensorPlan)
    CUDA.unsafe_free!(plan.workspace)
    if plan.handle != C_NULL
        unsafe_destroy!(plan)
        plan.handle = C_NULL
    end
end


## descriptor

mutable struct CuTensorDescriptor
    handle::cutensorTensorDescriptor_t
    # inner constructor handles creation and finalizer of the descriptor
    function CuTensorDescriptor(sz::Vector{Int64}, st::Vector{Int64}, eltype::DataType,
                                alignmentRequirement::UInt32=UInt32(128))
        desc = Ref{cutensorTensorDescriptor_t}()
        length(st) == (N = length(sz)) || throw(ArgumentError("size and stride vectors must have the same length"))
        cutensorCreateTensorDescriptor(handle(), desc, N, sz, st, eltype, alignmentRequirement)

        obj = new(desc[])
        finalizer(unsafe_destroy!, obj)
        return obj
    end
end

# outer constructor restricted to DenseCuArray, but could be extended
function CuTensorDescriptor(a::DenseCuArray; size=size(a), strides=strides(a), eltype=eltype(a))
    sz = collect(Int64, size)
    st = collect(Int64, strides)
    return CuTensorDescriptor(sz, st, eltype)
end

Base.show(io::IO, desc::CuTensorDescriptor) = @printf(io, "CuTensorDescriptor(%p)", desc.handle)

Base.unsafe_convert(::Type{cutensorTensorDescriptor_t}, obj::CuTensorDescriptor) = obj.handle

function unsafe_destroy!(obj::CuTensorDescriptor)
    cutensorDestroyTensorDescriptor(obj)
end


## tensor

export CuTensor

mutable struct CuTensor{T, N}
    data::CuArray{T, N}
    inds::Vector{Int32}

    function CuTensor{T, N}(data::CuArray{T,N}, inds::Vector) where {T<:Number, N}
        new(data, inds)
    end
end

CuTensor(data::CuArray{T,N}, inds::Vector) where {T<:Number, N} =
    CuTensor{T,N}(data, inds)

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
