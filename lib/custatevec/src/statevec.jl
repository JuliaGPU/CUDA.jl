function applyPauliExp!(sv::CuStateVec, theta::Float64, paulis::Vector{<:Pauli}, targets::Vector{Int32}, controls::Vector{Int32}, controlValues::Vector{Int32}=fill(one(Int32), length(controls)))
    cupaulis = CuStateVecPauli.(paulis)
    custatevecApplyPauliRotation(handle(), pointer(sv.data), sv.nbits, theta, cupaulis, targets, length(targets), controls, controlVals, length(controls))
    sv
end

function applyMatrix!(sv::CuStateVec, matrix::Union{Matrix, CuMatrix}, adjoint::Bool, targets::Vector{Int32}, controls::Vector{Int32}, controlValues::Vector{Int32}=fill(one(Int32), length(controls)))
    function bufferSize()
        out = Ref{Csize_t}()
        custatevecApplyMatrixGetWorkspaceSize(handle(), eltype(sv), sv.nbits, pointer(matrix), eltype(matrix), CUSTATEVEC_MATRIX_LAYOUT_COL, Int32(adjoint), length(targets), length(controls), compute_type(eltype(sv), eltype(matrix)), out)
        out[]
    end
    with_workspace(bufferSize) do buffer
        custatevecApplyMatrix(handle(), pointer(sv.data), eltype(sv), sv.nbits, pointer(matrix), eltype(matrix), CUSTATEVEC_MATRIX_LAYOUT_COL, Int32(adjoint), targets, length(targets), controls, controlValues, length(controls), compute_type(eltype(sv), eltype(matrix)), pointer(buffer), length(buffer))
    end
    sv
end

function expectation(sv::CuStateVec, matrix::Union{Matrix, CuMatrix}, basis_bits::Vector{Int32})
    function bufferSize()
        out = Ref{Csize_t}()
        custatevecComputeExpectationGetWorkspaceSize(handle(), eltype(sv), sv.nbits, pointer(matrix), eltype(matrix), CUSTATEVEC_MATRIX_LAYOUT_COL, length(basis_bits), compute_type(eltype(sv), eltype(matrix)), out)
        out[]
    end
    expVal = Ref{Float64}()
    residualNorm = Ref{Float64}()
    with_workspace(bufferSize) do buffer
        custatevecComputeExpectation(handle(), pointer(sv.data), eltype(sv), sv.nbits, expVal, Float64, residualNorm, pointer(matrix), eltype(matrix), CUSTATEVEC_MATRIX_LAYOUT_COL, basis_bits, length(basis_bits), compute_type(eltype(sv), eltype(matrix)), buffer, length(buffer))
    end
    return expVal[], residualNorm[]
end

function sample(sv::CuStateVec, sampled_bits::Vector{Int32}, shot_count)
    sampler = CuStateVecSampler(sv, UInt32(shot_count))
    bitstrings = Vector{custatevecIndex_t}(undef, shot_count)
    with_workspace(sampler.ws_size) do buffer
        custatevecSamplerPreprocess(handle(), Ref(sampler.handle), buffer, length(buffer))
        custatevecSamplerSample(handle(), Ref(sampler.handle), bitstrings, sampled_bits, length(sampled_bits), rand(shot_count), shot_count, CUSTATEVEC_SAMPLER_OUTPUT_RANDNUM_ORDER)
    end
    return bitstrings
end
