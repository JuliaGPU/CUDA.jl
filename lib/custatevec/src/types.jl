# cuStateVec types

## custatevec compute type

function Base.convert(::Type{custatevecComputeType_t}, T::DataType)
    if T == Float32
        return CUSTATEVEC_COMPUTE_32F
    elseif T == Float64
        return CUSTATEVEC_COMPUTE_64F
    else
        throw(ArgumentError("cuStateVec type equivalent for compute type $T does not exist!"))
    end
end

function Base.Type(T::custatevecComputeType_t)
    if T == CUSTATEVEC_COMPUTE_32F || T == CUSTATEVEC_COMPUTE_TF32
        return Float32
    elseif T == CUSTATEVEC_COMPUTE_64F
        return Float64
    else
        throw(ArgumentError("Julia type equivalent for compute type $T does not exist!"))
    end
end

function compute_type(sv_type::DataType, mat_type::DataType)
    if sv_type == ComplexF64 && mat_type == ComplexF64
        return Float64
    elseif sv_type == ComplexF32 && mat_type <: Union{ComplexF64, ComplexF32}
        return Float32
    end
end

abstract type Pauli end
struct PauliX <: Pauli end
struct PauliY <: Pauli end
struct PauliZ <: Pauli end
struct PauliI <: Pauli end
CuStateVecPauli(pauli::PauliX) = CUSTATEVEC_PAULI_X
CuStateVecPauli(pauli::PauliY) = CUSTATEVEC_PAULI_Y
CuStateVecPauli(pauli::PauliZ) = CUSTATEVEC_PAULI_Z
CuStateVecPauli(pauli::PauliI) = CUSTATEVEC_PAULI_I


mutable struct CuStateVec{T}
    data::CuVector{T}
    nbits::UInt32
end
function CuStateVec(T, n_qubits::Int; sv_type::custatevecStateVectorType_t=CUSTATEVEC_STATE_VECTOR_TYPE_ZERO)
    data = CuVector{T}(undef, 2^n_qubits)
    # in most cases, taking the hit here for setting one element
    # is cheaper than building the entire thing on the CPU and
    # copying it over
    sv = CuStateVec{T}(data, n_qubits)
    initialize!(sv, sv_type)
    return sv
end
CuStateVec(v::CuVector{T}) where {T} = CuStateVec{T}(v, UInt32(log2(length(v))))
CuStateVec(v::Vector{T}) where {T}   = CuStateVec(CuVector{T}(v))

Base.eltype(sv::CuStateVec{T}) where T = T
Base.copy(sv::CuStateVec{T}) where {T} = CuStateVec(copy(sv.data))

mutable struct CuStateVecSampler
    handle::custatevecSamplerDescriptor_t
    ws_size::Csize_t
    # the descriptor refers to the state vector's memory, and to the workspace once preprocessed
    sv::CuStateVec
    workspace::Union{Nothing,CuVector{UInt8}}
    function CuStateVecSampler(sv::CuStateVec, shot_count::UInt32)
        desc_ref   = Ref{custatevecSamplerDescriptor_t}()
        extra_size = Ref{Csize_t}(0)
        custatevecSamplerCreate(handle(), pointer(sv.data), eltype(sv), sv.nbits, desc_ref, shot_count, extra_size)
        obj = new(desc_ref[], extra_size[], sv, nothing)
        resource_finalizer(custatevecSamplerDestroy, obj)
        obj
    end
end

Base.unsafe_convert(::Type{custatevecSamplerDescriptor_t}, desc::CuStateVecSampler) = desc.handle

mutable struct CuStateVecAccessor
    handle::custatevecAccessorDescriptor_t
    ws_size::Csize_t
    # the descriptor refers to the state vector's memory, and to the extra workspace if set
    sv::CuStateVec
    workspace::Union{Nothing,CuVector{UInt8}}
    # with `view=true`, the accessor is read-only, but the state vector can be immutable
    function CuStateVecAccessor(sv::CuStateVec, bit_ordering::Vector{Int}, mask_bit_string::Vector{Int}, mask_ordering::Vector{Int}; view::Bool=false)
        desc_ref   = Ref{custatevecAccessorDescriptor_t}()
        extra_size = Ref{Csize_t}(0)
        mask_string = isempty(mask_bit_string) ? C_NULL : convert(Vector{Int32}, mask_bit_string)
        mask_order  = isempty(mask_ordering)   ? C_NULL : convert(Vector{Int32}, mask_ordering)
        create = view ? custatevecAccessorCreateView : custatevecAccessorCreate
        create(handle(), pointer(sv.data), eltype(sv), sv.nbits, desc_ref, convert(Vector{Int32}, bit_ordering), length(bit_ordering), mask_string, mask_order, length(mask_bit_string), extra_size)
        obj = new(desc_ref[], extra_size[], sv, nothing)
        resource_finalizer(custatevecAccessorDestroy, obj)
        # the library crashes when accessing the state vector without the workspace it asked for
        extra_size[] > 0 && setExtraWorkspace!(obj, CuVector{UInt8}(undef, extra_size[]))
        obj
    end
end

Base.unsafe_convert(::Type{custatevecAccessorDescriptor_t}, desc::CuStateVecAccessor) = desc.handle

mutable struct CuStateVecDistIndexBitSwapScheduler
    handle::custatevecDistIndexBitSwapSchedulerDescriptor_t
    function CuStateVecDistIndexBitSwapScheduler(n_global_bits::Integer, n_local_bits::Integer)
        desc_ref = Ref{custatevecDistIndexBitSwapSchedulerDescriptor_t}()
        custatevecDistIndexBitSwapSchedulerCreate(handle(), desc_ref, n_global_bits, n_local_bits)
        obj = new(desc_ref[])
        resource_finalizer(custatevecDistIndexBitSwapSchedulerDestroy, obj)
        obj
    end
end

Base.unsafe_convert(::Type{custatevecDistIndexBitSwapSchedulerDescriptor_t}, desc::CuStateVecDistIndexBitSwapScheduler) = desc.handle

mutable struct CuStateVecSubSVMigrator{T}
    handle::custatevecSubSVMigratorDescriptor_t
    # the descriptor refers to this memory
    device_slices::CuVector{T}
    function CuStateVecSubSVMigrator(device_slices::CuVector{T}, n_device_slices::Integer, n_local_bits::Integer) where {T}
        length(device_slices) == n_device_slices * 2^n_local_bits || throw(DimensionMismatch("device_slices must hold $n_device_slices slices of $(2^n_local_bits) elements."))
        desc_ref = Ref{custatevecSubSVMigratorDescriptor_t}()
        custatevecSubSVMigratorCreate(handle(), desc_ref, device_slices, T, n_device_slices, n_local_bits)
        obj = new{T}(desc_ref[], device_slices)
        resource_finalizer(custatevecSubSVMigratorDestroy, obj)
        obj
    end
end

Base.unsafe_convert(::Type{custatevecSubSVMigratorDescriptor_t}, desc::CuStateVecSubSVMigrator) = desc.handle
