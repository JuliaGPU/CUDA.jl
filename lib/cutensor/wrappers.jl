# wrappers of low-level functionality

function version()
    ver = cutensorGetVersion()
    major, ver = divrem(ver, 10000)
    minor, patch = divrem(ver, 100)

    VersionNumber(major, minor, patch)
end

function cuda_version()
  ver = cutensorGetCudartVersion()
  major, ver = divrem(ver, 1000)
  minor, patch = divrem(ver, 10)

  VersionNumber(major, minor, patch)
end

const ModeType = AbstractVector{<:Union{Char, Integer}}

is_unary(op::cutensorOperator_t) =
    (op ∈ (CUTENSOR_OP_IDENTITY, CUTENSOR_OP_SQRT, CUTENSOR_OP_RELU, CUTENSOR_OP_CONJ,
            CUTENSOR_OP_RCP))
is_binary(op::cutensorOperator_t) =
    (op ∈ (CUTENSOR_OP_ADD, CUTENSOR_OP_MUL, CUTENSOR_OP_MAX, CUTENSOR_OP_MIN))

mutable struct CuTensorDescriptor
    desc::Ref{cutensorTensorDescriptor_t}

    function CuTensorDescriptor(a; size = size(a), strides = strides(a), eltype = eltype(a),
                                   op = CUTENSOR_OP_IDENTITY)
        sz = collect(Int64, size)
        st = collect(Int64, strides)
        desc = Ref{cutensorTensorDescriptor_t}()
        cutensorInitTensorDescriptor(handle(), desc, length(sz), sz, st,
                                     eltype, op)
        obj = new(desc)
        return obj
    end
end

const scalar_types = Dict(
    (Float16, Float16)          => Float32,
    (Float32, Float16)          => Float32,
    (Float32, Float32)          => Float32,
    (Float64, Float64)          => Float64,
    (Float64, Float32)          => Float64,
    (ComplexF32, ComplexF32)    => ComplexF32,
    (ComplexF64, ComplexF64)    => ComplexF64,
    (ComplexF64, ComplexF32)    => ComplexF64)

Base.cconvert(::Type{Ptr{cutensorTensorDescriptor_t}}, obj::CuTensorDescriptor) = obj.desc

function elementwiseTrinary!(
        @nospecialize(alpha::Number),
        @nospecialize(A::DenseCuArray), Ainds::ModeType, opA::cutensorOperator_t,
        @nospecialize(beta::Number),
        @nospecialize(B::DenseCuArray), Binds::ModeType, opB::cutensorOperator_t,
        @nospecialize(gamma::Number),
        @nospecialize(C::DenseCuArray), Cinds::ModeType, opC::cutensorOperator_t,
        @nospecialize(D::DenseCuArray), Dinds::ModeType, opAB::cutensorOperator_t,
        opABC::cutensorOperator_t)
    !is_unary(opA)    && throw(ArgumentError("opA must be a unary op!"))
    !is_unary(opB)    && throw(ArgumentError("opB must be a unary op!"))
    !is_unary(opC)    && throw(ArgumentError("opC must be a unary op!"))
    !is_binary(opAB)  && throw(ArgumentError("opAB must be a binary op!"))
    !is_binary(opABC) && throw(ArgumentError("opABC must be a binary op!"))
    descA = CuTensorDescriptor(A; op = opA)
    descB = CuTensorDescriptor(B; op = opB)
    descC = CuTensorDescriptor(C; op = opC)
    @assert size(C) == size(D) && strides(C) == strides(D)
    descD = descC # must currently be identical
    modeA = collect(Cint, Ainds)
    modeB = collect(Cint, Binds)
    modeC = collect(Cint, Cinds)
    modeD = modeC
    scalar_type = scalar_types[(eltype(C), eltype(D))]
    cutensorElementwiseTrinary(handle(),
                                Ref{scalar_type}(alpha), A, descA, modeA,
                                Ref{scalar_type}(beta),  B, descB, modeB,
                                Ref{scalar_type}(gamma), C, descC, modeC,
                                D, descD, modeD,
                                opAB, opABC, scalar_type, stream())
    return D
end

function elementwiseTrinary!(
        @nospecialize(alpha::Number),
        @nospecialize(A::Array), Ainds::ModeType, opA::cutensorOperator_t,
        @nospecialize(beta::Number),
        @nospecialize(B::Array), Binds::ModeType, opB::cutensorOperator_t,
        @nospecialize(gamma::Number),
        @nospecialize(C::Array), Cinds::ModeType, opC::cutensorOperator_t,
        @nospecialize(D::Array), Dinds::ModeType, opAB::cutensorOperator_t,
        opABC::cutensorOperator_t)
    !is_unary(opA)    && throw(ArgumentError("opA must be a unary op!"))
    !is_unary(opB)    && throw(ArgumentError("opB must be a unary op!"))
    !is_unary(opC)    && throw(ArgumentError("opC must be a unary op!"))
    !is_binary(opAB)  && throw(ArgumentError("opAB must be a binary op!"))
    !is_binary(opABC) && throw(ArgumentError("opABC must be a binary op!"))
    descA = CuTensorDescriptor(A; op = opA)
    descB = CuTensorDescriptor(B; op = opB)
    descC = CuTensorDescriptor(C; op = opC)
    @assert size(C) == size(D) && strides(C) == strides(D)
    descD = descC # must currently be identical
    modeA = collect(Cint, Ainds)
    modeB = collect(Cint, Binds)
    modeC = collect(Cint, Cinds)
    modeD = modeC
    scalar_type = scalar_types[(eltype(C), eltype(D))]
    cutensorElementwiseTrinary(handle(),
                               Ref{scalar_type}(alpha), A, descA, modeA,
                               Ref{scalar_type}(beta),  B, descB, modeB,
                               Ref{scalar_type}(gamma), C, descC, modeC,
                               D, descD, modeD,
                               opAB, opABC, scalar_type, stream())
    return D
end

function elementwiseBinary!(
        @nospecialize(alpha::Number),
        @nospecialize(A::DenseCuArray), Ainds::ModeType, opA::cutensorOperator_t,
        @nospecialize(gamma::Number),
        @nospecialize(C::DenseCuArray), Cinds::ModeType, opC::cutensorOperator_t,
        @nospecialize(D::DenseCuArray), Dinds::ModeType, opAC::cutensorOperator_t)
    !is_unary(opA)    && throw(ArgumentError("opA must be a unary op!"))
    !is_unary(opC)    && throw(ArgumentError("opC must be a unary op!"))
    !is_binary(opAC)  && throw(ArgumentError("opAC must be a binary op!"))
    descA = CuTensorDescriptor(A; op = opA)
    descC = CuTensorDescriptor(C; op = opC)
    @assert size(C) == size(D) && strides(C) == strides(D)
    descD = descC # must currently be identical
    modeA = collect(Cint, Ainds)
    modeC = collect(Cint, Cinds)
    modeD = modeC
    scalar_type = scalar_types[(eltype(C), eltype(D))]
    cutensorElementwiseBinary(handle(),
                              Ref{scalar_type}(alpha), A, descA, modeA,
                              Ref{scalar_type}(gamma), C, descC, modeC,
                              D, descD, modeD,
                              opAC, scalar_type, stream())
    return D
end

function elementwiseBinary!(
        @nospecialize(alpha::Number),
        @nospecialize(A::Array), Ainds::ModeType, opA::cutensorOperator_t,
        @nospecialize(gamma::Number),
        @nospecialize(C::Array), Cinds::ModeType, opC::cutensorOperator_t,
        @nospecialize(D::Array), Dinds::ModeType, opAC::cutensorOperator_t)
    !is_unary(opA)    && throw(ArgumentError("opA must be a unary op!"))
    !is_unary(opC)    && throw(ArgumentError("opC must be a unary op!"))
    !is_binary(opAC)  && throw(ArgumentError("opAC must be a binary op!"))
    descA = CuTensorDescriptor(A; op = opA)
    descC = CuTensorDescriptor(C; op = opC)
    @assert size(C) == size(D) && strides(C) == strides(D)
    descD = descC # must currently be identical
    modeA = collect(Cint, Ainds)
    modeC = collect(Cint, Cinds)
    modeD = modeC
    scalar_type = scalar_types[(eltype(C), eltype(D))]
    cutensorElementwiseBinary(handle(),
                              Ref{scalar_type}(alpha), A, descA, modeA,
                              Ref{scalar_type}(gamma), C, descC, modeC,
                              D, descD, modeD,
                              opAC, scalar_type, stream())
    return D
end

function elementwiseBinary!(
        @nospecialize(alpha::Number),
        @nospecialize(A::CuTensor), opA::cutensorOperator_t,
        @nospecialize(gamma::Number),
        @nospecialize(C::CuTensor), opC::cutensorOperator_t,
        @nospecialize(D::CuTensor), opAC::cutensorOperator_t)
    !is_unary(opA)    && throw(ArgumentError("opA must be a unary op!"))
    !is_unary(opC)    && throw(ArgumentError("opC must be a unary op!"))
    !is_binary(opAC)  && throw(ArgumentError("opAC must be a binary op!"))
    descA = CuTensorDescriptor(A; op = opA)
    descC = CuTensorDescriptor(C; op = opC)
    @assert size(C) == size(D) && strides(C) == strides(D)
    descD = descC # must currently be identical
    modeA = collect(Cint, A.inds)
    modeC = collect(Cint, C.inds)
    modeD = modeC
    scalar_type = scalar_types[(eltype(C), eltype(D))]
    cutensorElementwiseBinary(handle(),
                              Ref{scalar_type}(alpha), A.data, descA, modeA,
                              Ref{scalar_type}(gamma), C.data, descC, modeC,
                              D.data, descD, modeD,
                              opAC, scalar_type, stream())
    return D
end

function permutation!(
        @nospecialize(alpha::Number),
        @nospecialize(A::DenseCuArray), Ainds::ModeType,
        @nospecialize(B::DenseCuArray), Binds::ModeType)
    #!is_unary(opPsi)    && throw(ArgumentError("opPsi must be a unary op!"))
    descA = CuTensorDescriptor(A)
    descB = CuTensorDescriptor(B)
    scalar_type = eltype(B)
    modeA = collect(Cint, Ainds)
    modeB = collect(Cint, Binds)
    cutensorPermutation(handle(),
                        Ref{scalar_type}(alpha),
                        A, descA, modeA,
                        B, descB, modeB,
                        scalar_type, stream())
    return B
end
function permutation!(
        @nospecialize(alpha::Number),
        @nospecialize(A::Array), Ainds::ModeType,
        @nospecialize(B::Array), Binds::ModeType)
    #!is_unary(opPsi)    && throw(ArgumentError("opPsi must be a unary op!"))
    descA = CuTensorDescriptor(A)
    descB = CuTensorDescriptor(B)
    scalar_type = eltype(B)
    modeA = collect(Cint, Ainds)
    modeB = collect(Cint, Binds)
    cutensorPermutation(handle(),
                        Ref{scalar_type}(alpha),
                        A, descA, modeA,
                        B, descB, modeB,
                        scalar_type, stream())
    return B
end

function contraction!(
        @nospecialize(alpha::Number),
        @nospecialize(A::Union{Array, CuArray}), Ainds::ModeType, opA::cutensorOperator_t,
        @nospecialize(B::Union{Array, CuArray}), Binds::ModeType, opB::cutensorOperator_t,
        @nospecialize(beta::Number),
        @nospecialize(C::Union{Array, CuArray}), Cinds::ModeType, opC::cutensorOperator_t,
        opOut::cutensorOperator_t;
        pref::cutensorWorksizePreference_t=CUTENSOR_WORKSPACE_RECOMMENDED,
        algo::cutensorAlgo_t=CUTENSOR_ALGO_DEFAULT,
        compute_type::Type=eltype(C), plan::Union{cutensorContractionPlan_t, Nothing}=nothing)
    !is_unary(opA)    && throw(ArgumentError("opA must be a unary op!"))
    !is_unary(opB)    && throw(ArgumentError("opB must be a unary op!"))
    !is_unary(opC)    && throw(ArgumentError("opC must be a unary op!"))
    !is_unary(opOut)  && throw(ArgumentError("opOut must be a unary op!"))
    descA = CuTensorDescriptor(A; op = opA)
    descB = CuTensorDescriptor(B; op = opB)
    descC = CuTensorDescriptor(C; op = opC)
    # for now, D must be identical to C (and thus, descD must be identical to descC)
    output_type = eltype(C)
    scalar_type = scalar_types[(output_type, compute_type)]
    computeType = cutensorComputeType(compute_type)
    modeA = collect(Cint, Ainds)
    modeB = collect(Cint, Binds)
    modeC = collect(Cint, Cinds)

    alignmentRequirementA = Ref{UInt32}(C_NULL)
    cutensorGetAlignmentRequirement(handle(), A, descA, alignmentRequirementA)
    alignmentRequirementB = Ref{UInt32}(C_NULL)
    cutensorGetAlignmentRequirement(handle(), B, descB, alignmentRequirementB)
    alignmentRequirementC = Ref{UInt32}(C_NULL)
    cutensorGetAlignmentRequirement(handle(), C, descC, alignmentRequirementC)
    desc = Ref{cutensorContractionDescriptor_t}()
    cutensorInitContractionDescriptor(handle(),
                                      desc,
                   descA, modeA, alignmentRequirementA[],
                   descB, modeB, alignmentRequirementB[],
                   descC, modeC, alignmentRequirementC[],
                   descC, modeC, alignmentRequirementC[],
                   computeType)
    find = Ref{cutensorContractionFind_t}()
    cutensorInitContractionFind(handle(), find, algo)

        function workspaceSize()
            @nospecialize
            out = Ref{UInt64}(C_NULL)
            cutensorContractionGetWorkspace(handle(), desc, find, pref, out)
            return out[]
        end
        with_workspace(workspaceSize, 1<<27) do workspace
            @nospecialize
            plan_ref = Ref{cutensorContractionPlan_t}()
            if isnothing(plan)
                cutensorInitContractionPlan(handle(), plan_ref, desc, find, sizeof(workspace))
            else
                plan_ref = Ref(plan)
            end
            cutensorContraction(handle(), plan_ref,
                                Ref{scalar_type}(alpha), A, B,
                                Ref{scalar_type}(beta),  C, C,
                                workspace, sizeof(workspace), stream())
        end
    return C
end

function plan_contraction(
        @nospecialize(A::Union{CuArray, Array}), Ainds::ModeType, opA::cutensorOperator_t,
        @nospecialize(B::Union{CuArray, Array}), Binds::ModeType, opB::cutensorOperator_t,
        @nospecialize(C::Union{CuArray, Array}), Cinds::ModeType, opC::cutensorOperator_t,
        opOut::cutensorOperator_t;
        pref::cutensorWorksizePreference_t=CUTENSOR_WORKSPACE_RECOMMENDED,
        algo::cutensorAlgo_t=CUTENSOR_ALGO_DEFAULT, compute_type::Type=eltype(C))
    !is_unary(opA)    && throw(ArgumentError("opA must be a unary op!"))
    !is_unary(opB)    && throw(ArgumentError("opB must be a unary op!"))
    !is_unary(opC)    && throw(ArgumentError("opC must be a unary op!"))
    !is_unary(opOut)  && throw(ArgumentError("opOut must be a unary op!"))
    descA = CuTensorDescriptor(A; op = opA)
    descB = CuTensorDescriptor(B; op = opB)
    descC = CuTensorDescriptor(C; op = opC)
    # for now, D must be identical to C (and thus, descD must be identical to descC)
    output_type = eltype(C)
    scalar_type = scalar_types[(output_type, compute_type)]
    computeType = cutensorComputeType(compute_type)
    modeA = collect(Cint, Ainds)
    modeB = collect(Cint, Binds)
    modeC = collect(Cint, Cinds)

    alignmentRequirementA = Ref{UInt32}(C_NULL)
    cutensorGetAlignmentRequirement(handle(), A, descA, alignmentRequirementA)
    alignmentRequirementB = Ref{UInt32}(C_NULL)
    cutensorGetAlignmentRequirement(handle(), B, descB, alignmentRequirementB)
    alignmentRequirementC = Ref{UInt32}(C_NULL)
    cutensorGetAlignmentRequirement(handle(), C, descC, alignmentRequirementC)
    desc = Ref{cutensorContractionDescriptor_t}()
    cutensorInitContractionDescriptor(handle(),
                                      desc,
                   descA, modeA, alignmentRequirementA[],
                   descB, modeB, alignmentRequirementB[],
                   descC, modeC, alignmentRequirementC[],
                   descC, modeC, alignmentRequirementC[],
                   computeType)

    find = Ref{cutensorContractionFind_t}()
    cutensorInitContractionFind(handle(), find, algo)
    plan = Ref{cutensorContractionPlan_t}()
    workspace_size = Ref{UInt64}(C_NULL)
    cutensorContractionGetWorkspace(handle(), desc, find, pref, workspace_size)
    cutensorInitContractionPlan(handle(), plan, desc, find, workspace_size[])
    return plan[]
end

function reduction!(
        @nospecialize(alpha::Number),
        @nospecialize(A::Union{Array, CuArray}), Ainds::ModeType, opA::cutensorOperator_t,
        @nospecialize(beta::Number),
        @nospecialize(C::Union{Array, CuArray}), Cinds::ModeType, opC::cutensorOperator_t,
        opReduce::cutensorOperator_t)
    !is_unary(opA)    && throw(ArgumentError("opA must be a unary op!"))
    !is_unary(opC)    && throw(ArgumentError("opC must be a unary op!"))
    !is_binary(opReduce)  && throw(ArgumentError("opReduce must be a binary op!"))
    descA = CuTensorDescriptor(A; op = opA)
    descC = CuTensorDescriptor(C; op = opC)
    # for now, D must be identical to C (and thus, descD must be identical to descC)
    T = eltype(C)
    typeCompute = cutensorComputeType(T)
    modeA = collect(Cint, Ainds)
    modeC = collect(Cint, Cinds)

    function workspaceSize()
        @nospecialize
        out = Ref{UInt64}(C_NULL)
        cutensorReductionGetWorkspace(handle(),
            A, descA, modeA,
            C, descC, modeC,
            C, descC, modeC,
            opReduce, typeCompute,
            out)
        return out[]
    end
    with_workspace(workspaceSize, 1<<13) do workspace
        @nospecialize
        cutensorReduction(handle(),
            Ref{T}(alpha), A, descA, modeA,
            Ref{T}(beta),  C, descC, modeC,
                    C, descC, modeC,
            opReduce, typeCompute,
            workspace, sizeof(workspace), stream())
    end

    return C
end

function cutensorComputeType(T::DataType)
    if T == Float32
        return CUTENSOR_COMPUTE_32F
    elseif T == ComplexF32
        return CUTENSOR_COMPUTE_32F
    elseif T == Float16
        return CUTENSOR_COMPUTE_16F
    elseif T == ComplexF16
        return CUTENSOR_COMPUTE_16F
    elseif T == Float64
        return CUTENSOR_COMPUTE_64F
    elseif T == ComplexF64
        return CUTENSOR_COMPUTE_64F
    elseif T == Int8
        return CUTENSOR_COMPUTE_8I
    elseif T == Int32
        return CUTENSOR_COMPUTE_32I
    elseif T == UInt8
        return CUTENSOR_COMPUTE_8U
    elseif T == UInt32
        return CUTENSOR_COMPUTE_32U
    else
        throw(ArgumentError("cutensorComputeType equivalent for input type $T does not exist!"))
    end
end
