# cuStateVec types

## custatevec compute type

function Base.convert(::Type{custatevecComputeType_t}, T::DataType)
    if T == Float16
        return CUSTATEVEC_COMPUTE_16F
    elseif T == Float32 
        return CUSTATEVEC_COMPUTE_32F
    elseif T == Float64
        return CUSTATEVEC_COMPUTE_64F
    elseif T == UInt8 
        return CUSTATEVEC_COMPUTE_8U
    elseif T == Int8 
        return CUSTATEVEC_COMPUTE_8I
    elseif T == UInt32 
        return CUSTATEVEC_COMPUTE_32U
    elseif T == Int32 
        return CUSTATEVEC_COMPUTE_32I
    else
        throw(ArgumentError("CUSTATEVEC type equivalent for compute type $T does not exist!"))
    end
end

function Base.convert(::Type{Type}, T::custatevecComputeType_t)
    if T == CUSTATEVEC_COMPUTE_16F
        return Float16 
    elseif T == CUSTATEVEC_COMPUTE_32F
        return Float32
    elseif T == CUSTATEVEC_COMPUTE_64F
        return Float64
    elseif T == CUSTATEVEC_COMPUTE_8U
        return UInt8 
    elseif T == CUSTATEVEC_COMPUTE_32U
        return UInt32
    elseif T == CUSTATEVEC_COMPUTE_8I
        return Int8 
    elseif T == CUSTATEVEC_COMPUTE_32I
        return Int32
    else
        throw(ArgumentError("Julia type equivalent for compute type $T does not exist!"))
    end
end

function compute_type(sv_type::DataType, mat_type::DataType)
    if sv_type == ComplexF64 && mat_type == ComplexF64
        return Float64
    elseif sv_type == ComplexF32 && mat_type == ComplexF64
        return Float32
    elseif sv_type == ComplexF32 && mat_type == ComplexF32
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
function CuStateVec(T, n_qubits::Int)
    data = CUDA.zeros(T, 2^n_qubits)
    # in most cases, taking the hit here for setting one element
    # is cheaper than building the entire thing on the CPU and 
    # copying it over
    CUDA.@allowscalar data[1] = one(T)
    CuStateVec{T}(data, n_qubits)
end

Base.eltype(sv::CuStateVec{T}) where T = T

mutable struct CuStateVecSampler
    handle::custatevecSamplerDescriptor_t
    ws_size::Csize_t
    function CuStateVecSampler(sv::CuStateVec, shot_count::UInt32)
        desc_ref   = Ref{custatevecSamplerDescriptor_t}()
        extra_size = Ref{Csize_t}()
        custatevecSampler_create(handle(), pointer(sv.data), eltype(sv), sv.nbits, desc_ref, shot_count, extra_size)
        obj = new(desc_ref[], extra_size[])
        #finalizer(custatevecSampler_destroy, obj) # apparently there is no such method?
        obj
    end
end

Base.unsafe_convert(::Type{custatevecSamplerDescriptor_t}, desc::CuStateVecSampler) = desc.handle
