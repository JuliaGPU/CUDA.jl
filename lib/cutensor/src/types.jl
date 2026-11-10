## data types

@public CuTensorPlan, CuTensorDescriptor, cutensorComputeDescriptorEnum,
        COMPUTE_DESC_16F, COMPUTE_DESC_32F, COMPUTE_DESC_TF32, COMPUTE_DESC_3XTF32,
        COMPUTE_DESC_64F, COMPUTE_DESC_16BF, COMPUTE_DESC_9X16BF, COMPUTE_DESC_8XINT8,
        COMPUTE_DESC_4X16F

@enum cutensorComputeDescriptorEnum begin
    COMPUTE_DESC_16F = 1
    COMPUTE_DESC_32F = 2
    COMPUTE_DESC_TF32 = 3
    COMPUTE_DESC_3XTF32 = 4
    COMPUTE_DESC_64F = 5
    COMPUTE_DESC_16BF = 6
    COMPUTE_DESC_9X16BF = 7
    COMPUTE_DESC_8XINT8 = 8
    COMPUTE_DESC_4X16F = 9
end

"""
    cutensorComputeDescriptorEnum

The precisions that cuTENSOR operations can compute in, for their `compute_type` keyword
argument: `COMPUTE_DESC_16F`, `COMPUTE_DESC_16BF`, `COMPUTE_DESC_TF32`,
`COMPUTE_DESC_3XTF32`, `COMPUTE_DESC_32F`, `COMPUTE_DESC_64F`, and the emulated
`COMPUTE_DESC_9X16BF`, `COMPUTE_DESC_8XINT8` and `COMPUTE_DESC_4X16F`. Passing a type
instead selects `COMPUTE_DESC_16F` for `Float16`, `COMPUTE_DESC_32F` for `Float32` and
`ComplexF32`, and `COMPUTE_DESC_64F` for `Float64` and `ComplexF64`. Which precisions an
operation supports depends on its element types, the GPU and the cuTENSOR version; see
NVIDIA's documentation.
"""
cutensorComputeDescriptorEnum

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
    (Float64,    Float64,    Float32)    => Float64,
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
    (Float32,    Float16)    => Float32,
    (Float32,    Float32)    => Float32,
    (Float64,    Float64)    => Float64,
    (Float32,    Float64)    => Float64,
    (Float64,    Float32)    => Float64,
    (ComplexF32, ComplexF32) => Float32,
    (ComplexF64, ComplexF64) => Float64,
    (ComplexF32, ComplexF64) => Float64,
    (ComplexF64, ComplexF32) => Float64,
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
    elseif T == COMPUTE_DESC_16BF
        return CUTENSOR_COMPUTE_DESC_16BF()
    elseif T == COMPUTE_DESC_9X16BF
        return CUTENSOR_COMPUTE_DESC_9X16BF()
    elseif T == COMPUTE_DESC_8XINT8
        return CUTENSOR_COMPUTE_DESC_8XINT8()
    elseif T == COMPUTE_DESC_4X16F
        return CUTENSOR_COMPUTE_DESC_4X16F()
    else
        throw(ArgumentError("cutensorComputeDescriptor equivalent for input enum value $T does not exist!"))
    end
end


# cutensorDataType_t is now just cudaDataType; conversions are defined in CUDA proper
# (lib/library_types.jl)


## plan

"""
    CuTensorPlan(desc::cutensorOperationDescriptor_t, pref::cutensorPlanPreference_t;
                 workspacePref=WORKSPACE_DEFAULT)

A cuTENSOR plan for one operation, together with the workspace it needs. The `plan_*`
functions ([`plan_contraction`](@ref), [`plan_permutation`](@ref), ...) return one, which
the operations ([`contract!`](@ref), [`cuTENSOR.permute!`](@ref), ...) execute without
planning again. Executions of a plan share its workspace, so don't execute one plan
concurrently on several streams, and use it in the CUDA context it was created in.

The constructor makes a plan for an operation descriptor and plan preference created with
the C API (`cutensorCreateContraction`, `cutensorCreatePlanPreference`, ...), which remain
owned by the caller. A plan converts to a `cutensorPlan_t`, and is freed when it is garbage
collected, or earlier with `CUDA.unsafe_free!(plan)`. Its fields are not part of the API.
"""
mutable struct CuTensorPlan
    ctx::CuContext
    handle::cutensorPlan_t
    workspace::CuVector{UInt8,CUDACore.DeviceMemory}
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

        obj = new(context(), plan_ref[], workspace, Type(required_scalar_type[]))
        resource_finalizer(CUDACore.unsafe_free!, obj)
        return obj
    end
end

Base.show(io::IO, plan::CuTensorPlan) = @printf(io, "CuTensorPlan(%p)", plan.handle)

Base.unsafe_convert(::Type{cutensorPlan_t}, plan::CuTensorPlan) = plan.handle

Base.:(==)(a::CuTensorPlan, b::CuTensorPlan) = a.handle == b.handle
Base.hash(plan::CuTensorPlan, h::UInt) = hash(plan.handle, h)

# destroying the plan
function unsafe_destroy!(plan::CuTensorPlan)
    context!(plan.ctx) do
        cutensorDestroyPlan(plan)
    end
end

# freeing the plan and associated workspace
function CUDACore.unsafe_free!(plan::CuTensorPlan)
    if plan.handle != C_NULL
        unsafe_destroy!(plan)
        plan.handle = C_NULL
    end
    CUDACore.unsafe_free!(plan.workspace)
end


const CUTENSOR_ALIGNMENT = UInt32(128)

## descriptor

"""
    CuTensorDescriptor(A::DenseCuArray; size=size(A), strides=strides(A), eltype=eltype(A))
    CuTensorDescriptor(size::Vector{Int64}, strides::Vector{Int64}, eltype::DataType,
                       alignment::UInt32=0x80)

A cuTENSOR tensor descriptor: the size, strides and element type of an array, and the
alignment in bytes of its data. The operations and `plan_*` functions describe their
arguments with `CuTensorDescriptor(A)`, so other array types can be supported by adding a
method to it. It converts to a `cutensorTensorDescriptor_t`, for use with the C API, and is
destroyed when it is garbage collected. Its fields are not part of the API.
"""
mutable struct CuTensorDescriptor
    handle::cutensorTensorDescriptor_t
    # inner constructor handles creation and finalizer of the descriptor
    function CuTensorDescriptor(sz::Vector{Int64}, st::Vector{Int64}, eltype::DataType,
                                alignmentRequirement::UInt32=CUTENSOR_ALIGNMENT)
        desc = Ref{cutensorTensorDescriptor_t}()
        length(st) == (N = length(sz)) || throw(ArgumentError("size and stride vectors must have the same length"))
        cutensorCreateTensorDescriptor(handle(), desc, N, sz, st, eltype, alignmentRequirement)

        obj = new(desc[])
        resource_finalizer(unsafe_destroy!, obj)
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

"""
    CuTensor(data::CuArray{T,N}, inds::Vector) -> CuTensor{T,N}

A dense tensor: the array `data` with a mode label for each of its dimensions, given as
`Char`s or integers in `inds`. Tensors with matching labels can be added and subtracted
(`+`, `-`, `axpy!`, `axpby!`), and multiplied (`*`, `mul!`), which contracts the modes
they share. `data` must be aligned to 128 bytes; otherwise it is copied, with a warning.
The fields `data` and `inds` are part of the API.
"""
mutable struct CuTensor{T, N}
    data::CuArray{T, N}
    inds::Vector{Int32}

    function CuTensor{T, N}(data::CuArray{T,N}, inds::Vector) where {T<:Number, N}
        if length(inds) != N
            throw(ArgumentError("The number of indices must match the number of dimensions of the data."))
        end
        if !iszero(UInt(pointer(data)) % CUTENSOR_ALIGNMENT)
            @warn "The data for this CuTensor does not obey the CUTENSOR alignment requirement of $CUTENSOR_ALIGNMENT. An explicit copy will be made to ensure the requirement is met."
            return new(copy(data), inds)
        else
            return new(data, inds)
        end
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
