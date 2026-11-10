@public initialize!, applyPauliRotation!, applyMatrix!, applyMatrixBatched!,
        applyGeneralizedPermutationMatrix!, abs2SumOnZBasis, collapseOnZBasis!,
        measureOnZBasis!, collapseByBitString!, collapseByBitStringBatched!, abs2SumArray,
        abs2SumArrayBatched, batchMeasure!, measureBatched!, batchMeasureWithOffset!,
        computeExpectation, computeExpectationBatched, computeExpectationsOnPauliBasis,
        sample, swapIndexBits!, multiDeviceSwapIndexBits!, testMatrixType, preprocess!,
        squaredNorm, applySubSVOffset!, setExtraWorkspace!, setIndexBitSwaps!,
        swapParameters, migrate!, accessorSet!, accessorGet

"""
    initialize!(sv::CuStateVec, sv_type::custatevecStateVectorType_t) -> sv

Set the state vector to a predefined state: `CUSTATEVEC_STATE_VECTOR_TYPE_ZERO` (``|0…0⟩``),
`_UNIFORM`, `_GHZ` or `_W`.

Wraps `custatevecInitializeStateVector`.
"""
function initialize!(sv::CuStateVec, sv_type::custatevecStateVectorType_t)
    custatevecInitializeStateVector(handle(), sv.data, eltype(sv), sv.nbits, sv_type)
    return sv
end

"""
    applyPauliRotation!(sv::CuStateVec, theta, paulis::Vector{<:Pauli}, targets, controls,
                        controlValues=ones(length(controls))) -> sv

Apply ``e^{iθP}`` to the state vector, where ``P`` is the product of `paulis` acting on the
`targets` bits, controlled by the `controls` bits having the `controlValues`.

Wraps `custatevecApplyPauliRotation`.
"""
function applyPauliRotation!(sv::CuStateVec, theta::Real, paulis::Vector{<:Pauli}, targets::Vector{<:Integer}, controls::Vector{<:Integer}, controlValues::Vector{<:Integer}=fill(one(Int32), length(controls)))
    cupaulis = CuStateVecPauli.(paulis)
    custatevecApplyPauliRotation(handle(), sv.data, eltype(sv), sv.nbits, theta, cupaulis, convert(Vector{Int32}, targets), length(targets), convert(Vector{Int32}, controls), convert(Vector{Int32}, controlValues), length(controls))
    return sv
end

"""
    applyMatrix!(sv::CuStateVec, matrix, adjoint::Bool, targets, controls,
                 controlValues=ones(length(controls))) -> sv

Apply the gate `matrix`, or its adjoint, to the `targets` bits of the state vector,
controlled by the `controls` bits having the `controlValues`. `matrix` is a `Matrix` or
`CuMatrix` of size `2^length(targets)`, whose row and column indices have the first target
as their least significant bit.

Wraps `custatevecApplyMatrix`.
"""
function applyMatrix!(sv::CuStateVec, matrix::Union{Matrix, CuMatrix}, adjoint::Bool, targets::Vector{<:Integer}, controls::Vector{<:Integer}, controlValues::Vector{<:Integer}=fill(one(Int32), length(controls)))
    function bufferSize()
        out = Ref{Csize_t}()
        custatevecApplyMatrixGetWorkspaceSize(handle(), eltype(sv), sv.nbits, matrix, eltype(matrix), CUSTATEVEC_MATRIX_LAYOUT_COL, Int32(adjoint), length(targets), length(controls), compute_type(eltype(sv), eltype(matrix)), out)
        out[]
    end
    with_workspace(handle().cache, bufferSize) do buffer
        custatevecApplyMatrix(handle(), sv.data, eltype(sv), sv.nbits, matrix, eltype(matrix), CUSTATEVEC_MATRIX_LAYOUT_COL, Int32(adjoint), convert(Vector{Int32}, targets), length(targets), convert(Vector{Int32}, controls), convert(Vector{Int32}, controlValues), length(controls), compute_type(eltype(sv), eltype(matrix)), buffer, sizeof(buffer))
    end
    return sv
end

"""
    applyMatrixBatched!(sv::CuStateVec, n_svs, map_type::custatevecMatrixMapType_t,
                        matrix_inds, matrices, n_matrices, adjoint::Bool, targets, controls,
                        controlValues=ones(length(controls))) -> sv

Apply gate matrices to each of the `n_svs` state vectors that `sv` holds one after the other,
as [`applyMatrix!`](@ref) does. `matrices` is a `Vector` or `CuVector` with the
`n_matrices` matrices one after the other. With `CUSTATEVEC_MATRIX_MAP_TYPE_BROADCAST`, the
first matrix is applied to every state vector, and with `_MATRIX_INDEXED`, the matrix with
the 0-based index `matrix_inds[i]` to state vector `i`.

Wraps `custatevecApplyMatrixBatched`.
"""
function applyMatrixBatched!(sv::CuStateVec, n_svs::Int, map_type::custatevecMatrixMapType_t, matrix_inds::Vector{Int}, matrix::Union{Vector, CuVector}, n_matrices::Int, adjoint::Bool, targets::Vector{<:Integer}, controls::Vector{<:Integer}, controlValues::Vector{<:Integer}=fill(one(Int32), length(controls)))
    sv_stride    = div(length(sv.data), n_svs)
    n_index_bits = Int(log2(div(length(sv.data), n_svs)))
    function bufferSize()
        out = Ref{Csize_t}()
        custatevecApplyMatrixBatchedGetWorkspaceSize(handle(), eltype(sv), n_index_bits, n_svs, sv_stride, map_type, convert(Vector{Int32}, matrix_inds), matrix, eltype(matrix), CUSTATEVEC_MATRIX_LAYOUT_COL, Int32(adjoint), n_matrices, length(targets), length(controls), compute_type(eltype(sv), eltype(matrix)), out)
        out[]
    end
    with_workspace(handle().cache, bufferSize) do buffer
        custatevecApplyMatrixBatched(handle(), sv.data, eltype(sv), n_index_bits, n_svs, sv_stride, map_type, convert(Vector{Int32}, matrix_inds), matrix, eltype(matrix), CUSTATEVEC_MATRIX_LAYOUT_COL, Int32(adjoint), n_matrices, convert(Vector{Int32}, targets), length(targets), convert(Vector{Int32}, controls), convert(Vector{Int32}, controlValues), length(controls), compute_type(eltype(sv), eltype(matrix)), buffer, sizeof(buffer))
    end
    return sv
end

index_vector(v::Vector{<:Integer}) = convert(Vector{custatevecIndex_t}, v)
index_vector(v::CuVector{<:Integer}) = convert(CuVector{custatevecIndex_t}, v)

"""
    applyGeneralizedPermutationMatrix!(sv::CuStateVec, permutation, diagonals, adjoint::Bool,
                                       targets, controls, controlValues=ones(length(controls)))
        -> sv

Apply the generalized permutation matrix ``DP``, or its adjoint, to the `targets` bits of the
state vector, controlled by the `controls` bits having the `controlValues`. ``P`` is given
by the 0-based `permutation` and ``D`` by its `diagonals`, both of length
`2^length(targets)`, in host or device memory.

Wraps `custatevecApplyGeneralizedPermutationMatrix`.
"""
function applyGeneralizedPermutationMatrix!(sv::CuStateVec, permutation::Union{Vector{<:Integer}, CuVector{<:Integer}}, diagonals::Union{Vector, CuVector}, adjoint::Bool, targets::Vector{<:Integer}, controls::Vector{<:Integer}, controlValues::Vector{<:Integer}=fill(one(Int32), length(controls)))
    permutation = index_vector(permutation)
    function bufferSize()
        out = Ref{Csize_t}()
        custatevecApplyGeneralizedPermutationMatrixGetWorkspaceSize(handle(), eltype(sv), sv.nbits, permutation, diagonals, eltype(diagonals), convert(Vector{Int32}, targets), length(targets), length(controls), out)
        out[]
    end
    with_workspace(handle().cache, bufferSize) do buffer
        custatevecApplyGeneralizedPermutationMatrix(handle(), sv.data, eltype(sv), sv.nbits, permutation, diagonals, eltype(diagonals), Int32(adjoint), convert(Vector{Int32}, targets), length(targets), convert(Vector{Int32}, controls), convert(Vector{Int32}, controlValues), length(controls), buffer, sizeof(buffer))
    end
    return sv
end

"""
    abs2SumOnZBasis(sv::CuStateVec, basisInds) -> abs2sum0, abs2sum1

The sums of the squared magnitudes of the elements of the state vector for which the `basisInds`
bits have even and odd parity. For a normalized state vector, these are the probabilities of
measuring either parity on that Z product basis.

Wraps `custatevecAbs2SumOnZBasis`.
"""
function abs2SumOnZBasis(sv::CuStateVec, basisInds::Vector{<:Integer})
    abs2sum0 = Ref{Float64}(0.0)
    abs2sum1 = Ref{Float64}(0.0)
    custatevecAbs2SumOnZBasis(handle(), sv.data, eltype(sv), sv.nbits, abs2sum0, abs2sum1, convert(Vector{Int32}, basisInds), length(basisInds))
    return abs2sum0[], abs2sum1[]
end

"""
    collapseOnZBasis!(sv::CuStateVec, parity, basisInds, norm) -> sv

Collapse the state vector on the Z product basis of the `basisInds` bits: scale the elements
for which these bits have the given `parity` by `1/sqrt(norm)`, and set the others to zero.
`norm` is typically the probability of that parity, from [`abs2SumOnZBasis`](@ref).

Wraps `custatevecCollapseOnZBasis`.
"""
function collapseOnZBasis!(sv::CuStateVec, parity::Int, basisInds::Vector{<:Integer}, norm::Float64)
    custatevecCollapseOnZBasis(handle(), sv.data, eltype(sv), sv.nbits, parity, convert(Vector{Int32}, basisInds), length(basisInds), norm)
    return sv
end

"""
    measureOnZBasis!(sv::CuStateVec, basisInds, randnum,
                     collapse=CUSTATEVEC_COLLAPSE_NONE) -> sv, parity

Measure the parity of the `basisInds` bits of the state vector, using the random number
`randnum` in `[0, 1)`. With `collapse = CUSTATEVEC_COLLAPSE_NORMALIZE_AND_ZERO`, the state
vector is also collapsed to the result.

Wraps `custatevecMeasureOnZBasis`.
"""
function measureOnZBasis!(sv::CuStateVec, basisInds::Vector{<:Integer}, randnum::Float64, collapse::custatevecCollapseOp_t=CUSTATEVEC_COLLAPSE_NONE)
    0.0 <= randnum < 1.0 || throw(ArgumentError("randnum $randnum must be in the interval [0, 1)."))
    parity = Ref{Int32}()
    custatevecMeasureOnZBasis(handle(), sv.data, eltype(sv), sv.nbits, parity, convert(Vector{Int32}, basisInds), length(basisInds), randnum, collapse)
    return sv, parity[]
end

"""
    collapseByBitString!(sv::CuStateVec, bitstring, bitordering, norm) -> sv

Collapse the state vector to the elements whose `bitordering` bits have the values in
`bitstring`, scaling them by `1/sqrt(norm)` and setting the others to zero.

Wraps `custatevecCollapseByBitString`.
"""
function collapseByBitString!(sv::CuStateVec, bitstring::Union{Vector{<:Integer}, BitVector, Vector{Bool}}, bitordering::Vector{<:Integer}, norm::Float64)
    custatevecCollapseByBitString(handle(), sv.data, eltype(sv), sv.nbits, convert(Vector{Int32}, bitstring), convert(Vector{Int32}, bitordering), length(bitstring), norm)
    return sv
end

"""
    collapseByBitStringBatched!(sv::CuStateVec, n_svs, bitstrings, bitordering, norms) -> sv

Collapse each of the `n_svs` state vectors that `sv` holds one after the other, as
[`collapseByBitString!`](@ref) does. The bit strings are given as integers in `bitstrings`,
and the normalization constants in `norms`.

Wraps `custatevecCollapseByBitStringBatched`.
"""
function collapseByBitStringBatched!(sv::CuStateVec, n_svs::Int, bitstrings::Vector{<:Integer}, bitordering::Vector{<:Integer}, norms::Vector{Float64})
    function bufferSize()
        out = Ref{Csize_t}()
        custatevecCollapseByBitStringBatchedGetWorkspaceSize(handle(), n_svs, convert(Vector{custatevecIndex_t}, bitstrings), norms, out)
        out[]
    end
    sv_stride    = div(length(sv.data), n_svs)
    n_index_bits = Int(log2(div(length(sv.data), n_svs)))
    with_workspace(handle().cache, bufferSize) do buffer
        custatevecCollapseByBitStringBatched(handle(), sv.data, eltype(sv), n_index_bits, n_svs, sv_stride, convert(Vector{custatevecIndex_t}, bitstrings), convert(Vector{Int32}, bitordering), n_index_bits, norms, buffer, sizeof(buffer))
    end
    return sv
end

"""
    abs2SumArray(sv::CuStateVec, bitordering, maskBitString, maskOrdering) -> Vector{Float64}

The sums of the squared magnitudes of the elements of the state vector for each value of the
`bitordering` bits (their probabilities, for a normalized state vector). The index of the
result has the first of these bits as its least significant bit. Only the elements whose
`maskOrdering` bits have the values in `maskBitString` are included.

Wraps `custatevecAbs2SumArray`.
"""
function abs2SumArray(sv::CuStateVec, bitordering::Vector{<:Integer}, maskBitString::Vector{<:Integer}, maskOrdering::Vector{<:Integer})
    abs2sum = Vector{Float64}(undef, 2^length(bitordering))
    custatevecAbs2SumArray(handle(), sv.data, eltype(sv), sv.nbits, abs2sum, convert(Vector{Int32}, bitordering), length(bitordering), convert(Vector{Int32}, maskBitString), convert(Vector{Int32}, maskOrdering), length(maskOrdering))
    return abs2sum
end

"""
    abs2SumArrayBatched(sv::CuStateVec, n_svs, bitordering, maskBitStrings, maskOrdering)
        -> Vector{Float64}

Compute [`abs2SumArray`](@ref) for each of the `n_svs` state vectors that `sv` holds one
after the other, with a mask bit string given as an integer in `maskBitStrings` for each.
The results are concatenated.

Wraps `custatevecAbs2SumArrayBatched`.
"""
function abs2SumArrayBatched(sv::CuStateVec, n_svs::Int, bitordering::Vector{<:Integer}, maskBitStrings::Vector{<:Integer}, maskOrdering::Vector{<:Integer})
    abs2sum      = zeros(Float64, n_svs * 2^length(bitordering))
    sv_stride    = div(length(sv.data), n_svs)
    n_index_bits = Int(log2(div(length(sv.data), n_svs)))
    sum_stride   = 2^length(bitordering)
    custatevecAbs2SumArrayBatched(handle(), sv.data, eltype(sv), n_index_bits, n_svs, sv_stride, abs2sum, sum_stride, convert(Vector{Int32}, bitordering), length(bitordering), convert(Vector{Int32}, maskBitStrings), convert(Vector{Int32}, maskOrdering), length(maskOrdering))
    return abs2sum
end

"""
    batchMeasure!(sv::CuStateVec, bitordering, randnum, collapse=CUSTATEVEC_COLLAPSE_NONE)
        -> sv, bitstring::Vector{Int32}

Measure the `bitordering` bits of the state vector, using the random number `randnum` in
`[0, 1)`, and return the result in the order of `bitordering`. With
`collapse = CUSTATEVEC_COLLAPSE_NORMALIZE_AND_ZERO`, the state vector is also collapsed to
the result.

Wraps `custatevecBatchMeasure`.
"""
function batchMeasure!(sv::CuStateVec, bitordering::Vector{<:Integer}, randnum::Float64, collapse::custatevecCollapseOp_t=CUSTATEVEC_COLLAPSE_NONE)
    0.0 <= randnum < 1.0 || throw(ArgumentError("randnum $randnum must be in the interval [0, 1)."))
    bitstring = zeros(Int32, length(bitordering))
    custatevecBatchMeasure(handle(), sv.data, eltype(sv), sv.nbits, bitstring, convert(Vector{Int32}, bitordering), length(bitstring), randnum, collapse)
    return sv, bitstring
end

"""
    measureBatched!(sv::CuStateVec, n_svs, bitordering, randnums,
                    collapse=CUSTATEVEC_COLLAPSE_NONE) -> sv, bitstrings::Vector{Int64}

Measure the `bitordering` bits of each of the `n_svs` state vectors that `sv` holds one after
the other, using one random number in `[0, 1)` for each. The results are returned as
integers whose least significant bit is the first of `bitordering`.

Wraps `custatevecMeasureBatched`.
"""
function measureBatched!(sv::CuStateVec, n_svs::Int, bitordering::Vector{<:Integer}, randnums::Vector{<:Real}, collapse::custatevecCollapseOp_t=CUSTATEVEC_COLLAPSE_NONE)
    length(randnums) == n_svs || throw(ArgumentError("randnums must have length $n_svs."))
    all(0.0 .<= randnums .< 1.0) || throw(ArgumentError("all randnums must be in the interval [0, 1)."))
    bitstrings   = zeros(custatevecIndex_t, n_svs)
    sv_stride    = div(length(sv.data), n_svs)
    n_index_bits = Int(log2(sv_stride))
    custatevecMeasureBatched(handle(), sv.data, eltype(sv), n_index_bits, n_svs, sv_stride, bitstrings, convert(Vector{Int32}, bitordering), length(bitordering), convert(Vector{Cdouble}, randnums), collapse)
    return sv, bitstrings
end

"""
    batchMeasureWithOffset!(sv::CuStateVec, bitordering, randnum, offset, abs2sum,
                            collapse=CUSTATEVEC_COLLAPSE_NONE) -> sv, bitstring::Vector{Int32}

Measure as [`batchMeasure!`](@ref) does, for a state vector that is part of a larger one
distributed over several devices or processes. `offset` is the sum of the squared magnitudes
of the parts before this one, and `abs2sum` that of the whole state vector.

Wraps `custatevecBatchMeasureWithOffset`.
"""
function batchMeasureWithOffset!(sv::CuStateVec, bitordering::Vector{<:Integer}, randnum::Float64, offset::Float64, abs2sum::Float64, collapse::custatevecCollapseOp_t=CUSTATEVEC_COLLAPSE_NONE)
    0.0 <= randnum < 1.0 || throw(ArgumentError("randnum $randnum must be in the interval [0, 1)."))
    bitstring = zeros(Int32, length(bitordering))
    custatevecBatchMeasureWithOffset(handle(), sv.data, eltype(sv), sv.nbits, convert(Vector{Int32}, bitstring), convert(Vector{Int32}, bitordering), length(bitstring), randnum, collapse, offset, abs2sum)
    return sv, bitstring
end

"""
    computeExpectation(sv::CuStateVec, matrix, basis_bits) -> Float64

The expectation value of the Hermitian `matrix` (a `Matrix` or `CuMatrix`) acting on the
`basis_bits` of the state vector. This waits for the result.

Wraps `custatevecComputeExpectation`.
"""
function computeExpectation(sv::CuStateVec, matrix::Union{Matrix, CuMatrix}, basis_bits::Vector{<:Integer})
    CT = compute_type(eltype(sv), eltype(matrix))
    function bufferSize()
        out = Ref{Csize_t}()
        custatevecComputeExpectationGetWorkspaceSize(handle(), eltype(sv), sv.nbits, matrix, eltype(matrix), CUSTATEVEC_MATRIX_LAYOUT_COL, length(basis_bits), CT, out)
        out[]
    end
    expVal = Ref{Float64}(0.0)
    residualNorm = Ref{Float64}(0.0)
    with_workspace(handle().cache, bufferSize) do buffer
        custatevecComputeExpectation(handle(), sv.data, eltype(sv), sv.nbits, expVal, Float64, residualNorm, matrix, eltype(matrix), CUSTATEVEC_MATRIX_LAYOUT_COL, convert(Vector{Int32}, basis_bits), length(basis_bits), CT, buffer, sizeof(buffer))
        synchronize()
    end
    return expVal[]
end

"""
    computeExpectationBatched(sv::CuStateVec, n_svs, matrices, n_matrices, basis_bits)
        -> Matrix{ComplexF64}

The expectation values of several matrices for several state vectors, as an
`n_matrices×n_svs` matrix. `matrices` holds the `n_matrices` matrices one after the other,
and `sv` the `n_svs` state vectors. This waits for the result.

Wraps `custatevecComputeExpectationBatched`.
"""
function computeExpectationBatched(sv::CuStateVec, n_svs::Int, matrices::Union{Vector, CuVector}, n_matrices::Int, basis_bits::Vector{<:Integer})
    sv_stride    = div(length(sv.data), n_svs)
    n_index_bits = Int(log2(sv_stride))
    CT = compute_type(eltype(sv), eltype(matrices))
    function bufferSize()
        out = Ref{Csize_t}()
        custatevecComputeExpectationBatchedGetWorkspaceSize(handle(), eltype(sv), n_index_bits, n_svs, sv_stride, matrices, eltype(matrices), CUSTATEVEC_MATRIX_LAYOUT_COL, n_matrices, length(basis_bits), CT, out)
        out[]
    end
    expVals = zeros(ComplexF64, n_matrices * n_svs)
    with_workspace(handle().cache, bufferSize) do buffer
        custatevecComputeExpectationBatched(handle(), sv.data, eltype(sv), n_index_bits, n_svs, sv_stride, expVals, matrices, eltype(matrices), CUSTATEVEC_MATRIX_LAYOUT_COL, n_matrices, convert(Vector{Int32}, basis_bits), length(basis_bits), CT, buffer, sizeof(buffer))
        synchronize()
    end
    # the C API documents the output as (nMatrices × nSVs) with leading
    # dimension nMatrices, i.e. matrix index is fastest-varying.
    return reshape(expVals, n_matrices, n_svs)
end

"""
    computeExpectationsOnPauliBasis(sv::CuStateVec, pauliOps, basisInds) -> Vector{Float64}

The expectation values of products of Pauli matrices: `pauliOps[i]` is a vector of
[`Pauli`](@ref)s acting on the bits `basisInds[i]` of the state vector.

Wraps `custatevecComputeExpectationsOnPauliBasis`.
"""
function computeExpectationsOnPauliBasis(sv::CuStateVec, pauliOps::AbstractVector{<:AbstractVector{<:Pauli}}, basisInds::AbstractVector{<:AbstractVector{<:Integer}})
    length(pauliOps) == length(basisInds) || throw(DimensionMismatch("need as many sets of basis bits as Pauli products."))
    all(i -> length(pauliOps[i]) == length(basisInds[i]), eachindex(pauliOps, basisInds)) || throw(DimensionMismatch("each Pauli product needs a basis bit for every Pauli matrix."))
    exp_vals = zeros(Float64, length(pauliOps))
    cupaulis = [[CuStateVecPauli(O) for O in op] for op in pauliOps]
    custatevecComputeExpectationsOnPauliBasis(handle(), sv.data, eltype(sv), sv.nbits, exp_vals, cupaulis, length(pauliOps), [convert(Vector{Int32}, b) for b in basisInds], convert(Vector{UInt32}, length.(basisInds)))
    return exp_vals
end

"""
    sample([rng,] sv::CuStateVec, sampled_bits, shot_count) -> Vector{Int64}
    sample(sampler::CuStateVecSampler, sampled_bits, randnums;
           output=CUSTATEVEC_SAMPLER_OUTPUT_RANDNUM_ORDER) -> Vector{Int64}

Sample the values of the `sampled_bits` of the state vector, as integers whose least
significant bit is the first sampled bit. The first form draws `shot_count` samples, with
random numbers from `rng` (by default the task-local generator). The second one uses a
sampler prepared with [`preprocess!`](@ref), and a sample for each of the given random
numbers in `[0, 1)`; `output` selects whether the samples are returned in the order of
the random numbers or in ascending order.

Wraps `custatevecSamplerSample`.
"""
sample(sv::CuStateVec, sampled_bits::Vector{<:Integer}, shot_count) =
    sample(default_rng(), sv, sampled_bits, shot_count)

function sample(rng::AbstractRNG, sv::CuStateVec, sampled_bits::Vector{<:Integer}, shot_count)
    sampler = CuStateVecSampler(sv, UInt32(shot_count))
    bitstrings = Vector{custatevecIndex_t}(undef, shot_count)
    with_workspace(handle().cache, sampler.ws_size) do buffer
        custatevecSamplerPreprocess(handle(), sampler, buffer, sizeof(buffer))
        custatevecSamplerSample(handle(), sampler, bitstrings, convert(Vector{Int32}, sampled_bits), length(sampled_bits), rand(rng, shot_count), shot_count, CUSTATEVEC_SAMPLER_OUTPUT_RANDNUM_ORDER)
    end
    return bitstrings
end

"""
    swapIndexBits!(sv::CuStateVec, bitSwaps::Vector{<:Pair}, maskBitString, maskOrdering)
        -> sv

Swap pairs of index bits of the state vector, i.e. reorder its elements. Only the elements
whose `maskOrdering` bits have the values in `maskBitString` are swapped.

Wraps `custatevecSwapIndexBits`.
"""
function swapIndexBits!(sv::CuStateVec, bitSwaps::Vector{Pair{T, T}}, maskBitString::Vector{<:Integer}, maskOrdering::Vector{<:Integer}) where {T<:Integer}
    custatevecSwapIndexBits(handle(), sv.data, eltype(sv), sv.nbits, convert(Vector{Pair{Int32, Int32}}, bitSwaps), length(bitSwaps), convert(Vector{Int32}, maskBitString), convert(Vector{Int32}, maskOrdering), length(maskOrdering))
    return sv
end

"""
    multiDeviceSwapIndexBits!(sub_svs::Vector{<:CuStateVec}, devices::Vector{CuDevice},
                              indexBitSwaps::Vector{<:Pair}, maskBitString, maskOrdering,
                              device_network_type::custatevecDeviceNetworkType_t) -> sub_svs

Swap pairs of index bits of a state vector that is distributed over the sub state vectors
`sub_svs`, which reside on the corresponding `devices`, one of which must be the current one.
Their number is a power of two, and the global index bits that select the sub state vector
follow the local ones. Only the elements whose `maskOrdering` bits have the values in
`maskBitString` are swapped.

NVIDIA requires peer access to be enabled in both directions between all these devices
(with `CUDACore.enable_peer_access`), and this crashes otherwise.

Wraps `custatevecMultiDeviceSwapIndexBits`.
"""
function multiDeviceSwapIndexBits!(sub_svs::Vector{<:CuStateVec}, devices::Vector{CuDevice}, indexBitSwaps::Vector{Pair{T, T}}, maskBitString::Vector{<:Integer}, maskOrdering::Vector{<:Integer}, device_network_type::custatevecDeviceNetworkType_t) where {T<:Integer}
    all(sv->sv.nbits == first(sub_svs).nbits, sub_svs) || throw(ArgumentError("all sub-vectors of the state vector must have the same number of index bits."))
    all(sv->eltype(sv) == eltype(first(sub_svs)), sub_svs) || throw(ArgumentError("all sub-vectors of the state vector must have the same element type."))
    n_global_bits = trailing_zeros(length(sub_svs))
    ispow2(length(sub_svs)) || throw(ArgumentError("the number of sub state vectors must be a power of 2."))
    # the handles must include one for the current device, on whose stream the swap is ordered
    device() in devices || throw(ArgumentError("one of the devices must be the current one."))
    original_device = device()
    handles = map(unique(devices)) do d
        device!(d)
        handle()
    end
    device!(original_device)
    raw_handles = [h.handle for h in handles]
    sub_data = map(sv->pointer(sv.data), sub_svs)
    GC.@preserve sub_svs handles begin
        custatevecMultiDeviceSwapIndexBits(raw_handles, length(raw_handles), sub_data, eltype(first(sub_svs)), n_global_bits, first(sub_svs).nbits, convert(Vector{Pair{Int32, Int32}}, indexBitSwaps), length(indexBitSwaps), convert(Vector{Int32}, maskBitString), convert(Vector{Int32}, maskOrdering), length(maskOrdering), device_network_type)
    end
    return sub_svs
end

"""
    testMatrixType(matrix, adjoint::Bool, matrix_type=CUSTATEVEC_MATRIX_TYPE_GENERAL,
                   compute_type=...) -> Float64

How far `matrix` (or its adjoint) is from being of `matrix_type`
(`CUSTATEVEC_MATRIX_TYPE_UNITARY` or `_HERMITIAN`), as a residual norm.

Wraps `custatevecTestMatrixType`.
"""
function testMatrixType(matrix::Union{Matrix, CuMatrix}, adjoint::Bool, matrix_type::custatevecMatrixType_t=CUSTATEVEC_MATRIX_TYPE_GENERAL, compute_type::custatevecComputeType_t=convert(custatevecComputeType_t, real(eltype(matrix))))
    n, m = size(matrix)
    n == m || throw(DimensionMismatch("matrix must be square, but has dimensions ($n, $m)."))
    n_targets = log2(n)
    n_targets > 15 && throw(ArgumentError("matrix must be smaller than 2^15 x 2^15"))

    residualNorm = Ref{Float64}()
    function bufferSize()
        out = Ref{Csize_t}()
        custatevecTestMatrixTypeGetWorkspaceSize(handle(), matrix_type, matrix, eltype(matrix), CUSTATEVEC_MATRIX_LAYOUT_COL, n_targets, Int32(adjoint), compute_type, out)
        out[]
    end
    with_workspace(handle().cache, bufferSize) do buffer
        custatevecTestMatrixType(handle(), residualNorm, matrix_type, matrix, eltype(matrix), CUSTATEVEC_MATRIX_LAYOUT_COL, n_targets, Int32(adjoint), compute_type, buffer, sizeof(buffer))
    end
    return residualNorm[]
end

"""
    preprocess!(sampler::CuStateVecSampler) -> sampler

Prepare the sampler for [`sample`](@ref), allocating its workspace.

Wraps `custatevecSamplerPreprocess`.
"""
function preprocess!(sampler::CuStateVecSampler)
    sampler.workspace = CuVector{UInt8}(undef, sampler.ws_size)
    custatevecSamplerPreprocess(handle(), sampler, sampler.workspace, sizeof(sampler.workspace))
    return sampler
end

"""
    squaredNorm(sampler::CuStateVecSampler) -> Float64

The squared norm of the sampler's state vector. The sampler must have been prepared with
[`preprocess!`](@ref).

Wraps `custatevecSamplerGetSquaredNorm`.
"""
function squaredNorm(sampler::CuStateVecSampler)
    norm = Ref{Cdouble}()
    custatevecSamplerGetSquaredNorm(handle(), sampler, norm)
    return norm[]
end

"""
    applySubSVOffset!(sampler::CuStateVecSampler, sub_sv_ord, n_sub_svs, offset, norm)
        -> sampler

Treat the sampler's state vector as part of a larger one. It is the 0-based part
`sub_sv_ord` of `n_sub_svs` parts, `offset` is the squared norm of the preceding parts, and
`norm` that of all parts. The sampler must have been prepared with [`preprocess!`](@ref); random
numbers given to [`sample`](@ref) that fall outside this part are clamped to it.

Wraps `custatevecSamplerApplySubSVOffset`.
"""
function applySubSVOffset!(sampler::CuStateVecSampler, sub_sv_ord::Integer, n_sub_svs::Integer, offset::Real, norm::Real)
    custatevecSamplerApplySubSVOffset(handle(), sampler, sub_sv_ord, n_sub_svs, offset, norm)
    return sampler
end

function sample(sampler::CuStateVecSampler, sampled_bits::Vector{<:Integer}, randnums::Vector{Float64}; output::custatevecSamplerOutput_t=CUSTATEVEC_SAMPLER_OUTPUT_RANDNUM_ORDER)
    bitstrings = Vector{custatevecIndex_t}(undef, length(randnums))
    custatevecSamplerSample(handle(), sampler, bitstrings, convert(Vector{Int32}, sampled_bits), length(sampled_bits), randnums, length(randnums), output)
    return bitstrings
end

function defaultWorkspaceSize()
    size = Ref{Csize_t}()
    custatevecGetDefaultWorkspaceSize(handle(), size)
    return size[]
end

# set through `CUDACore.math_mode!`
function mathMode()
    mode = Ref{custatevecMathMode_t}()
    custatevecGetMathMode(handle(), mode)
    return mode[]
end

function property(type::libraryPropertyType)
    value = Ref{Int32}()
    custatevecGetProperty(type, value)
    return Int(value[])
end

# the raw stream that's set on the library handle
function currentStream()
    stream_ref = Ref{cudaStream_t}()
    custatevecGetStream(handle(), stream_ref)
    return stream_ref[]
end

errorName(status::custatevecStatus_t)   = unsafe_string(custatevecGetErrorName(status))
errorString(status::custatevecStatus_t) = unsafe_string(custatevecGetErrorString(status))

"""
    setExtraWorkspace!(a::CuStateVecAccessor, buffer::CuVector{UInt8}) -> a

Give the accessor an extra workspace, of at least `a.ws_size` bytes, which speeds up
repeated accesses with small buffers. The accessor keeps it alive.

Wraps `custatevecAccessorSetExtraWorkspace`.
"""
function setExtraWorkspace!(a::CuStateVecAccessor, buffer::CuVector{UInt8})
    sizeof(buffer) >= a.ws_size || throw(ArgumentError("the accessor needs a workspace of at least $(a.ws_size) bytes."))
    custatevecAccessorSetExtraWorkspace(handle(), a, buffer, sizeof(buffer))
    a.workspace = buffer
    return a
end

"""
    setIndexBitSwaps!(scheduler::CuStateVecDistIndexBitSwapScheduler, bit_swaps::Vector{<:Pair},
                      mask_bit_string=Int32[], mask_ordering=Int32[]) -> Int

Plan the swaps of index bits of a distributed state vector, and return the number of swap
batches needed. Query each batch with [`swapParameters`](@ref).

Wraps `custatevecDistIndexBitSwapSchedulerSetIndexBitSwaps`.
"""
function setIndexBitSwaps!(scheduler::CuStateVecDistIndexBitSwapScheduler, bit_swaps::Vector{Pair{T, T}}, mask_bit_string::Vector{<:Integer}=Int32[], mask_ordering::Vector{<:Integer}=Int32[]) where {T<:Integer}
    n_batches = Ref{UInt32}()
    custatevecDistIndexBitSwapSchedulerSetIndexBitSwaps(handle(), scheduler, convert(Vector{Pair{Int32, Int32}}, bit_swaps), length(bit_swaps), convert(Vector{Int32}, mask_bit_string), convert(Vector{Int32}, mask_ordering), length(mask_ordering), n_batches)
    return Int(n_batches[])
end

"""
    swapParameters(scheduler::CuStateVecDistIndexBitSwapScheduler, swap_batch_index,
                   org_sub_sv_index) -> custatevecSVSwapParameters_t

The parameters of the data transfer of the 0-based sub state vector `org_sub_sv_index` in
the swap batch `swap_batch_index`, planned with [`setIndexBitSwaps!`](@ref).

Wraps `custatevecDistIndexBitSwapSchedulerGetParameters`.
"""
function swapParameters(scheduler::CuStateVecDistIndexBitSwapScheduler, swap_batch_index::Integer, org_sub_sv_index::Integer)
    parameters = Ref{custatevecSVSwapParameters_t}()
    custatevecDistIndexBitSwapSchedulerGetParameters(handle(), scheduler, swap_batch_index, org_sub_sv_index, parameters)
    return parameters[]
end

"""
    migrate!(migrator::CuStateVecSubSVMigrator{T}, slice_index, src, dst, i_begin, i_end)
        -> migrator

Copy the elements `[i_begin, i_end)` (0-based) of the sub state vector `src` into the device
slice `slice_index`, and/or those of the slice out to `dst`; with both, they are exchanged.
`src` and `dst` are host vectors, or `nothing`, in memory the device can access (e.g. pinned
with `CUDACore.pin`). The copies are asynchronous: synchronize before using or freeing that
memory.

Wraps `custatevecSubSVMigratorMigrate`.
"""
function migrate!(migrator::CuStateVecSubSVMigrator{T}, slice_index::Integer,
                  src::Union{Nothing, DenseVector{T}}, dst::Union{Nothing, DenseVector{T}},
                  i_begin::Integer, i_end::Integer) where {T}
    0 <= slice_index < migrator.n_device_slices || throw(ArgumentError("slice index $slice_index is out of range for $(migrator.n_device_slices) device slices."))
    0 <= i_begin <= i_end <= 2^migrator.n_local_bits || throw(ArgumentError("range [$i_begin, $i_end) is out of bounds for slices of $(2^migrator.n_local_bits) elements."))
    for buf in (src, dst)
        buf === nothing || length(buf) >= i_end || throw(DimensionMismatch("sub state vectors must hold at least $i_end elements."))
    end
    custatevecSubSVMigratorMigrate(handle(), migrator, slice_index, something(src, C_NULL), something(dst, C_NULL), i_begin, i_end)
    return migrator
end

"""
    accessorSet!(a::CuStateVecAccessor, buffer, i_begin, i_end)

Set the elements `[i_begin, i_end)` (0-based, in the accessor's bit ordering) of the state
vector from `buffer`. That is a `Vector` or `CuVector` with the element type of the state
vector and at least `i_end - i_begin` elements; its size and type are not checked.

Wraps `custatevecAccessorSet`.
"""
function accessorSet!(a::CuStateVecAccessor, external_buf::Union{Vector, CuVector}, i_begin::Int, i_end::Int)
    custatevecAccessorSet(handle(), a, pointer(external_buf), i_begin, i_end)
end

"""
    accessorGet(a::CuStateVecAccessor, buffer, i_begin, i_end)

Copy the elements `[i_begin, i_end)` (0-based, in the accessor's bit ordering) of the state
vector into `buffer`. That is a `Vector` or `CuVector` with the element type of the state
vector and at least `i_end - i_begin` elements; its size and type are not checked.

Wraps `custatevecAccessorGet`.
"""
function accessorGet(a::CuStateVecAccessor, external_buf::Union{Vector, CuVector}, i_begin::Int, i_end::Int)
    custatevecAccessorGet(handle(), a, pointer(external_buf), i_begin, i_end)
end
