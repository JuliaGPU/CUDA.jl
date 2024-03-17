function applyPauliExp!(sv::CuStateVec, theta::Float64, paulis::Vector{<:Pauli}, targets::Vector{Int32}, controls::Vector{Int32}, controlValues::Vector{Int32}=fill(one(Int32), length(controls)))
    cupaulis = CuStateVecPauli.(paulis)
    custatevecApplyPauliRotation(handle(), sv.data, eltype(sv), sv.nbits, theta, cupaulis, targets, length(targets), controls, controlValues, length(controls))
    sv
end

function applyMatrix!(sv::CuStateVec, matrix::Union{Matrix, CuMatrix}, adjoint::Bool, targets::Vector{<:Integer}, controls::Vector{<:Integer}, controlValues::Vector{<:Integer}=fill(one(Int32), length(controls)))
    function bufferSize()
        out = Ref{Csize_t}()
        custatevecApplyMatrixGetWorkspaceSize(handle(), eltype(sv), sv.nbits, matrix, eltype(matrix), CUSTATEVEC_MATRIX_LAYOUT_COL, Int32(adjoint), length(targets), length(controls), compute_type(eltype(sv), eltype(matrix)), out)
        out[]
    end
    with_workspace(handle().cache, bufferSize) do buffer
        custatevecApplyMatrix(handle(), sv.data, eltype(sv), sv.nbits, matrix, eltype(matrix), CUSTATEVEC_MATRIX_LAYOUT_COL, Int32(adjoint), convert(Vector{Int32}, targets), length(targets), convert(Vector{Int32}, controls), convert(Vector{Int32}, controlValues), length(controls), compute_type(eltype(sv), eltype(matrix)), buffer, length(buffer))
    end
    sv
end

function applyMatrixBatched!(sv::CuStateVec, n_svs::Int, map_type::custatevecMatrixMapType_t, matrix_inds::Vector{Int}, matrix::Union{Vector, CuVector}, n_matrices::Int, adjoint::Bool, targets::Vector{<:Integer}, controls::Vector{<:Integer}, controlValues::Vector{<:Integer}=fill(one(Int32), length(controls)))
    sv_stride    = div(length(sv.data), n_svs)
    n_index_bits = Int(log2(div(length(sv.data), n_svs)))
    function bufferSize()
        out = Ref{Csize_t}()
        custatevecApplyMatrixBatchedGetWorkspaceSize(handle(), eltype(sv), n_index_bits, n_svs, sv_stride, map_type, matrix_inds, matrix, eltype(matrix), CUSTATEVEC_MATRIX_LAYOUT_COL, Int32(adjoint), n_matrices, length(targets), length(controls), compute_type(eltype(sv), eltype(matrix)), out)
        out[]
    end
    with_workspace(handle().cache, bufferSize) do buffer
        custatevecApplyMatrixBatched(handle(), sv.data, eltype(sv), n_index_bits, n_svs, sv_stride, map_type, matrix_inds, matrix, eltype(matrix), CUSTATEVEC_MATRIX_LAYOUT_COL, Int32(adjoint), n_matrices, convert(Vector{Int32}, targets), length(targets), convert(Vector{Int32}, controls), convert(Vector{Int32}, controlValues), length(controls), compute_type(eltype(sv), eltype(matrix)), buffer, length(buffer))
    end
    sv
end

function applyGeneralizedPermutationMatrix!(sv::CuStateVec, permutation::Union{Vector{<:Integer}, CuVector{<:Integer}}, diagonals::Union{Vector, CuVector}, adjoint::Bool, targets::Vector{<:Integer}, controls::Vector{<:Integer}, controlValues::Vector{<:Integer}=fill(one(Int32), length(controls)))
    function bufferSize()
        out = Ref{Csize_t}()
        custatevecApplyGeneralizedPermutationMatrixGetWorkspaceSize(handle(), eltype(sv), sv.nbits, permutation, diagonals, eltype(diagonals), convert(Vector{Int32}, targets), length(targets), length(controls), out)
        out[]
    end
    with_workspace(handle().cache, bufferSize) do buffer
        custatevecApplyGeneralizedPermutationMatrix(handle(), sv.data, eltype(sv), sv.nbits, permutation, diagonals, eltype(diagonals), Int32(adjoint), convert(Vector{Int32}, targets), length(targets), convert(Vector{Int32}, controls), convert(Vector{Int32}, controlValues), length(controls), buffer, length(buffer))
    end
    sv
end

function abs2SumOnZBasis(sv::CuStateVec, basisInds::Vector{<:Integer})
    abs2sum0 = Ref{Float64}(0.0)
    abs2sum1 = Ref{Float64}(0.0)
    custatevecAbs2SumOnZBasis(handle(), sv.data, eltype(sv), sv.nbits, abs2sum0, abs2sum1, basisInds, length(basisInds))
    return abs2sum0[], abs2sum1[]
end

function collapseOnZBasis!(sv::CuStateVec, parity::Int, basisInds::Vector{<:Integer}, norm::Float64)
    custatevecCollapseOnZBasis(handle(), sv.data, eltype(sv), sv.nbits, parity, convert(Vector{Int32}, basisInds), length(basisInds), norm)
    sv
end

function measureOnZBasis!(sv::CuStateVec, basisInds::Vector{<:Integer}, randnum::Float64, collapse::custatevecCollapseOp_t=CUSTATEVEC_COLLAPSE_NONE)
    0.0 <= randnum < 1.0 || throw(ArgumentError("randnum $randnum must be in the interval [0, 1)."))
    parity = Ref{Int32}()
    custatevecMeasureOnZBasis(handle(), sv.data, eltype(sv), sv.nbits, parity, basisInds, length(basisInds), randnum, collapse)
    return sv, parity[]
end

function collapseByBitString!(sv::CuStateVec, bitstring::Union{Vector{<:Integer}, BitVector, Vector{Bool}}, bitordering::Vector{<:Integer}, norm::Float64)
    custatevecCollapseByBitString(handle(), sv.data, eltype(sv), sv.nbits, convert(Vector{Int32}, bitstring), convert(Vector{Int32}, bitordering), length(bitstring), norm)
    sv
end

function collapseByBitStringBatched!(sv::CuStateVec, n_svs::Int, bitstrings::Vector{<:Integer}, bitordering::Vector{<:Integer}, norms::Vector{Float64})
    function bufferSize()
        out = Ref{Csize_t}()
        custatevecCollapseByBitStringBatchedGetWorkspaceSize(handle(), n_svs, convert(Vector{custatevecIndex_t}, bitstrings), norms, out)
        out[]
    end
    sv_stride    = div(length(sv.data), n_svs)
    n_index_bits = Int(log2(div(length(sv.data), n_svs)))
    with_workspace(handle().cache, bufferSize) do buffer
        custatevecCollapseByBitStringBatched(handle(), sv.data, eltype(sv), n_index_bits, n_svs, sv_stride, convert(Vector{custatevecIndex_t}, bitstrings), convert(Vector{Int32}, bitordering), n_index_bits, norms, buffer, length(buffer))
    end
    sv
end

function abs2SumArray(sv::CuStateVec, bitordering::Vector{<:Integer}, maskBitString::Vector{<:Integer}, maskOrdering::Vector{<:Integer})
    abs2sum = Vector{Float64}(undef, 2^length(bitordering))
    custatevecAbs2SumArray(handle(), sv.data, eltype(sv), sv.nbits, abs2sum, convert(Vector{Int32}, bitordering), length(bitordering), convert(Vector{Int32}, maskBitString), convert(Vector{Int32}, maskOrdering), length(maskOrdering))
    return abs2sum
end

function abs2SumArrayBatched(sv::CuStateVec, n_svs::Int, bitordering::Vector{<:Integer}, maskBitStrings::Vector{<:Integer}, maskOrdering::Vector{<:Integer})
    abs2sum      = zeros(Float64, n_svs * 2^length(bitordering))
    sv_stride    = div(length(sv.data), n_svs)
    n_index_bits = Int(log2(div(length(sv.data), n_svs)))
    sum_stride   = 2^length(bitordering)
    custatevecAbs2SumArrayBatched(handle(), sv.data, eltype(sv), n_index_bits, n_svs, sv_stride, abs2sum, sum_stride, convert(Vector{Int32}, bitordering), length(bitordering), convert(Vector{Int32}, maskBitStrings), convert(Vector{Int32}, maskOrdering), length(maskOrdering))
    return abs2sum
end

function batchMeasure!(sv::CuStateVec, bitordering::Vector{<:Integer}, randnum::Float64, collapse::custatevecCollapseOp_t=CUSTATEVEC_COLLAPSE_NONE)
    0.0 <= randnum < 1.0 || throw(ArgumentError("randnum $randnum must be in the interval [0, 1)."))
    bitstring = zeros(Int32, length(bitordering))
    custatevecBatchMeasure(handle(), sv.data, eltype(sv), sv.nbits, bitstring, convert(Vector{Int32}, bitordering), length(bitstring), randnum, collapse)
    return sv, bitstring
end

function batchMeasureWithOffset!(sv::CuStateVec, bitordering::Vector{<:Integer}, randnum::Float64, offset::Float64, abs2sum::Float64, collapse::custatevecCollapseOp_t=CUSTATEVEC_COLLAPSE_NONE)
    0.0 <= randnum < 1.0 || throw(ArgumentError("randnum $randnum must be in the interval [0, 1)."))
    bitstring = zeros(Int32, length(bitordering))
    custatevecBatchMeasure(handle(), sv.data, eltype(sv), sv.nbits, convert(Vector{Int32}, bitstring), convert(Vector{Int32}, bitordering), length(bitstring), randnum, collapse, offset, abs2sum)
    return sv, bitstring
end

function expectation(sv::CuStateVec, matrix::Union{Matrix, CuMatrix}, basis_bits::Vector{<:Integer})
    function bufferSize()
        out = Ref{Csize_t}()
        custatevecComputeExpectationGetWorkspaceSize(handle(), eltype(sv), sv.nbits, matrix, eltype(matrix), CUSTATEVEC_MATRIX_LAYOUT_COL, length(basis_bits), compute_type(eltype(sv), eltype(matrix)), out)
        out[]
    end
    expVal = Ref{Float64}()
    residualNorm = Ref{Float64}()
    with_workspace(handle().cache, bufferSize) do buffer
        custatevecComputeExpectation(handle(), sv.data, eltype(sv), sv.nbits, expVal, Float64, residualNorm, matrix, eltype(matrix), CUSTATEVEC_MATRIX_LAYOUT_COL, convert(Vector{Int32}, basis_bits), length(basis_bits), compute_type(eltype(sv), eltype(matrix)), buffer, length(buffer))
    end
    return expVal[], residualNorm[]
end

function expectationsOnPauliBasis(sv::CuStateVec, pauliOps::Vector{Vector{Pauli}}, basisInds::Vector{Vector{Int}})
    exp_vals = zeros(Float64, length(pauliOps))
    cupaulis = [[CuStateVecPauli(O) for O in op] for op in pauliOps]
    custatevecComputeExpectationsOnPauliBasis(handle(), sv.data, eltype(sv), sv.nbits, exp_vals, cupaulis, length(pauliOps), convert(Vector{Vector{Int32}}, basisInds), length.(basisInds))
    return exp_vals
end

function sample(sv::CuStateVec, sampled_bits::Vector{<:Integer}, shot_count)
    sampler = CuStateVecSampler(sv, UInt32(shot_count))
    bitstrings = Vector{custatevecIndex_t}(undef, shot_count)
    with_workspace(handle().cache, sampler.ws_size) do buffer
        custatevecSamplerPreprocess(handle(), sampler.handle, buffer, length(buffer))
        custatevecSamplerSample(handle(), sampler.handle, bitstrings, convert(Vector{Int32}, sampled_bits), length(sampled_bits), rand(shot_count), shot_count, CUSTATEVEC_SAMPLER_OUTPUT_RANDNUM_ORDER)
    end
    return bitstrings
end

function swapIndexBits!(sv::CuStateVec, bitSwaps::Vector{Pair{T, T}}, maskBitString::Vector{<:Integer}, maskOrdering::Vector{<:Integer}) where {T<:Integer}
    custatevecSwapIndexBits(handle(), sv.data, eltype(sv), sv.nbits, convert(Vector{Pair{Int32, Int32}}, bitSwaps), length(bitSwaps), convert(Vector{Int32}, maskBitString), convert(Vector{Int32}, maskOrdering), length(maskOrdering))
    sv
end

function swapIndexBitsMultiDevice!(sub_svs::Vector{CuStateVec}, devices::Vector{CuDevice}, indexBitSwaps::Vector{Pair{T, T}}, maskBitString::Vector{<:Integer}, maskOrdering::Vector{<:Integer}, device_network_type::custatevecDeviceNetworkType_t) where {T<:Integer}
    all(sv->sv.nbits == first(sub_svs).nbits, sub_svs) || throw(ArgumentError("all sub-vectors of the state vector must have the same number of index bits."))
    all(sv->eltype(sv) == eltype(first(sub_svs)), sub_svs) || throw(ArgumentError("all sub-vectors of the state vector must have the same element type."))
    original_device = device()
    handles = map(devices) do d
        device!(d)
        handle()
    end
    device!(original_device)
    sub_data = map(sv->sv.data, sub_svs)
    global_index_bits = mapreduce(sv->sv.nbits, +, sub_svs)
    custatevecMultiDeviceSwapIndexBits(handles, length(handles), sub_data, eltype(first(sub_svs)), first(sub_svs).nbits, global_index_bits, convert(Vector{Pair{Int32, Int32}}, indexBitSwaps), length(indexBitSwaps), convert(Vector{Int32}, maskBitString), convert(Vector{Int32}, maskOrdering), length(maskOrdering), device_network_type)
    return sub_svs
end

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
        custatevecTestMatrixType(handle(), residualNorm, matrix_type, matrix, eltype(matrix), CUSTATEVEC_MATRIX_LAYOUT_COL, n_targets, Int32(adjoint), compute_type, buffer, length(buffer))
    end
    return residualNorm[]
end

function accessorSet(a::CuStateVecAccessor, external_buf::Union{Vector, CuVector}, i_begin::Int, i_end::Int)
    custatevecAccessorSet(handle(), a, external_buf, i_begin, i_end)
end

function accessorGet(a::CuStateVecAccessor, external_buf::Union{Vector, CuVector}, i_begin::Int, i_end::Int)
    custatevecAccessorGet(handle(), a, external_buf, i_begin, i_end)
end
