# Deprecated in v6.6: wrappers are named after the cuStateVec routine they call.

@deprecate applyPauliExp!(sv::CuStateVec, theta::Float64, paulis::Vector{<:Pauli}, targets::Vector{Int32}, controls::Vector{Int32}) applyPauliRotation!(sv, theta, paulis, targets, controls) false
@deprecate applyPauliExp!(sv::CuStateVec, theta::Float64, paulis::Vector{<:Pauli}, targets::Vector{Int32}, controls::Vector{Int32}, controlValues::Vector{Int32}) applyPauliRotation!(sv, theta, paulis, targets, controls, controlValues) false
@deprecate expectationBatched(sv::CuStateVec, n_svs::Int, matrices::Union{Vector, CuVector}, n_matrices::Int, basis_bits::Vector{<:Integer}) computeExpectationBatched(sv, n_svs, matrices, n_matrices, basis_bits) false
@deprecate expectationsOnPauliBasis(sv::CuStateVec, pauliOps::Vector{Vector{Pauli}}, basisInds::Vector{Vector{Int}}) computeExpectationsOnPauliBasis(sv, pauliOps, basisInds) false
@deprecate swapIndexBitsMultiDevice!(sub_svs::Vector{<:CuStateVec}, devices::Vector{CuDevice}, indexBitSwaps::Vector{Pair{T, T}}, maskBitString::Vector{<:Integer}, maskOrdering::Vector{<:Integer}, device_network_type::custatevecDeviceNetworkType_t) where {T<:Integer} multiDeviceSwapIndexBits!(sub_svs, devices, indexBitSwaps, maskBitString, maskOrdering, device_network_type) false

# `expectation` returned the expectation value wrapped in a `Ref`, together with one for the
# residual norm, which cuStateVec doesn't compute and leaves at 0
function expectation(sv::CuStateVec, matrix::Union{Matrix, CuMatrix}, basis_bits::Vector{<:Integer})
    Base.depwarn("`expectation` is deprecated, use `computeExpectation`, which returns the expectation value instead of `Ref`s.", :expectation)
    return Ref(computeExpectation(sv, matrix, basis_bits)), Ref(0.0)
end

# Deprecated in v6.6: the element type is a type parameter, as for other arrays.
@deprecate CuStateVec(T, n_qubits::Int; kwargs...) CuStateVec{T}(n_qubits; kwargs...) false
