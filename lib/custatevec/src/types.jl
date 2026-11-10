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

@public CuStateVec, Pauli, PauliX, PauliY, PauliZ, PauliI, CuStateVecSampler,
        CuStateVecAccessor, CuStateVecDistIndexBitSwapScheduler, CuStateVecSubSVMigrator

"""
    Pauli

The Pauli matrices `PauliX()`, `PauliY()` and `PauliZ()`, and the identity `PauliI()`, for
[`applyPauliRotation!`](@ref) and [`computeExpectationsOnPauliBasis`](@ref).
"""
abstract type Pauli end
"""
    PauliX()

The Pauli X matrix, see [`Pauli`](@ref).
"""
struct PauliX <: Pauli end
"""
    PauliY()

The Pauli Y matrix, see [`Pauli`](@ref).
"""
struct PauliY <: Pauli end
"""
    PauliZ()

The Pauli Z matrix, see [`Pauli`](@ref).
"""
struct PauliZ <: Pauli end
"""
    PauliI()

The identity, see [`Pauli`](@ref).
"""
struct PauliI <: Pauli end
CuStateVecPauli(pauli::PauliX) = CUSTATEVEC_PAULI_X
CuStateVecPauli(pauli::PauliY) = CUSTATEVEC_PAULI_Y
CuStateVecPauli(pauli::PauliZ) = CUSTATEVEC_PAULI_Z
CuStateVecPauli(pauli::PauliI) = CUSTATEVEC_PAULI_I


"""
    CuStateVec{T}(n_qubits; sv_type=CUSTATEVEC_STATE_VECTOR_TYPE_ZERO)
    CuStateVec(data::CuVector{T})
    CuStateVec(data::Vector{T})

A state vector with elements of type `T`, `ComplexF32` or `ComplexF64`. The first form
creates one of `n_qubits` qubits in the state selected by `sv_type` (see
[`initialize!`](@ref)). The others take the amplitudes, whose number is a power of 2: a
`CuVector` is used directly, a `Vector` is copied.

The amplitudes are in the field `data`, a `CuVector{T}`, and the number of qubits in
`nbits`. Bit `i` (0-based) of the index of `data` is qubit `i`.
"""
mutable struct CuStateVec{T}
    data::CuVector{T}
    nbits::UInt32
end
function CuStateVec{T}(n_qubits::Integer; sv_type::custatevecStateVectorType_t=CUSTATEVEC_STATE_VECTOR_TYPE_ZERO) where {T}
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

"""
    CuStateVecSampler(sv::CuStateVec, shot_count::UInt32)

A sampler of the state vector, for up to `shot_count` samples at a time. Prepare it with
[`preprocess!`](@ref), then draw samples with [`sample`](@ref). The sampler uses the memory of
the state vector, which must not change in between.

Wraps `custatevecSamplerDescriptor_t`.
"""
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

"""
    CuStateVecAccessor(sv::CuStateVec, bit_ordering::Vector{Int}, mask_bit_string::Vector{Int},
                       mask_ordering::Vector{Int}; view=false)

An accessor that copies elements between the state vector and other buffers, with
[`accessorGet`](@ref) and [`accessorSet!`](@ref). The elements are numbered by the
`bit_ordering` bits, the first being the least significant, among the elements whose
`mask_ordering` bits have the values in `mask_bit_string`. With `view = true`, the accessor
can only read.

Wraps `custatevecAccessorDescriptor_t`.
"""
mutable struct CuStateVecAccessor
    handle::custatevecAccessorDescriptor_t
    ws_size::Csize_t
    # the descriptor refers to the state vector's memory, and to the extra workspace if set.
    # that workspace is optional, and speeds up repeated accesses with small device buffers.
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
        obj
    end
end

Base.unsafe_convert(::Type{custatevecAccessorDescriptor_t}, desc::CuStateVecAccessor) = desc.handle

"""
    CuStateVecDistIndexBitSwapScheduler(n_global_bits, n_local_bits)

A scheduler for swapping index bits of a state vector distributed over `2^n_global_bits`
sub state vectors of `n_local_bits` index bits each, for example over several processes.
See [`setIndexBitSwaps!`](@ref) and [`swapParameters`](@ref).

Wraps `custatevecDistIndexBitSwapSchedulerDescriptor_t`.
"""
mutable struct CuStateVecDistIndexBitSwapScheduler
    handle::custatevecDistIndexBitSwapSchedulerDescriptor_t
    # the library handle it was created with, which is also needed to destroy it
    library::Handle
    function CuStateVecDistIndexBitSwapScheduler(n_global_bits::Integer, n_local_bits::Integer)
        desc_ref = Ref{custatevecDistIndexBitSwapSchedulerDescriptor_t}()
        library = handle()
        custatevecDistIndexBitSwapSchedulerCreate(library, desc_ref, n_global_bits, n_local_bits)
        obj = new(desc_ref[], library)
        resource_finalizer(obj) do x
            custatevecDistIndexBitSwapSchedulerDestroy(x.library, x)
        end
        obj
    end
end

Base.unsafe_convert(::Type{custatevecDistIndexBitSwapSchedulerDescriptor_t}, desc::CuStateVecDistIndexBitSwapScheduler) = desc.handle

"""
    CuStateVecSubSVMigrator(device_slices::CuVector{T}, n_device_slices, n_local_bits)

A migrator of sub state vectors of `n_local_bits` index bits between host memory and
`n_device_slices` slices of `device_slices`, with [`migrate!`](@ref). The migrator uses the
memory of `device_slices` directly.

Wraps `custatevecSubSVMigratorDescriptor_t`.
"""
mutable struct CuStateVecSubSVMigrator{T}
    handle::custatevecSubSVMigratorDescriptor_t
    # the library handle it was created with, which is also needed to destroy it
    library::Handle
    # the descriptor refers to this memory
    device_slices::CuVector{T}
    n_device_slices::Int
    n_local_bits::Int
    function CuStateVecSubSVMigrator(device_slices::CuVector{T}, n_device_slices::Integer, n_local_bits::Integer) where {T}
        length(device_slices) >= n_device_slices * 2^n_local_bits || throw(DimensionMismatch("device_slices must hold $n_device_slices slices of $(2^n_local_bits) elements."))
        desc_ref = Ref{custatevecSubSVMigratorDescriptor_t}()
        library = handle()
        custatevecSubSVMigratorCreate(library, desc_ref, device_slices, T, n_device_slices, n_local_bits)
        obj = new{T}(desc_ref[], library, device_slices, n_device_slices, n_local_bits)
        resource_finalizer(obj) do x
            custatevecSubSVMigratorDestroy(x.library, x)
        end
        obj
    end
end

Base.unsafe_convert(::Type{custatevecSubSVMigratorDescriptor_t}, desc::CuStateVecSubSVMigrator) = desc.handle
