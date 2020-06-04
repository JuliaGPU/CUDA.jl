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
                                     cudaDataType(eltype), op)
        obj = new(desc)
        return obj
    end
end

const scalar_types = Dict((Float16, Float16)=>Float32, (Float32, Float16)=>Float32, (Float32, Float32)=>Float32, (Float64, Float64)=>Float64, (Float64, Float32)=>Float64, (ComplexF32, ComplexF32)=>ComplexF32, (ComplexF64, ComplexF64)=>ComplexF64, (ComplexF64, ComplexF32)=>ComplexF64)

Base.cconvert(::Type{Ptr{cutensorTensorDescriptor_t}}, obj::CuTensorDescriptor) = obj.desc

function elementwiseTrinary!(
    alpha::Number, A::CuArray, Ainds::ModeType, opA::cutensorOperator_t,
    beta::Number,  B::CuArray, Binds::ModeType, opB::cutensorOperator_t,
    gamma::Number, C::CuArray{T}, Cinds::ModeType, opC::cutensorOperator_t,
    D::CuArray{T}, Dinds::ModeType, opAB::cutensorOperator_t,
    opABC::cutensorOperator_t; stream::CuStream=CuDefaultStream()) where {T}

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
    typeCompute = cudaDataType(T)
    modeA = collect(Cint, Ainds)
    modeB = collect(Cint, Binds)
    modeC = collect(Cint, Cinds)
    modeD = modeC
    cutensorElementwiseTrinary(handle(),
                                T[alpha], A, descA, modeA,
                                T[beta],  B, descB, modeB,
                                T[gamma], C, descC, modeC,
                                          D, descD, modeD,
                                opAB, opABC, typeCompute, stream)
    return D
end

function elementwiseTrinary!(
    alpha::Number, A::Array, Ainds::ModeType, opA::cutensorOperator_t,
    beta::Number, B::Array, Binds::ModeType, opB::cutensorOperator_t,
    gamma::Number, C::Array{T}, Cinds::ModeType, opC::cutensorOperator_t,
    D::Array{T}, Dinds::ModeType, opAB::cutensorOperator_t,
    opABC::cutensorOperator_t; stream::CuStream=CuDefaultStream()) where {T}

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
    typeCompute = cudaDataType(T)
    modeA = collect(Cint, Ainds)
    modeB = collect(Cint, Binds)
    modeC = collect(Cint, Cinds)
    modeD = modeC
    cutensorElementwiseTrinary(handle(),
                               T[alpha], A, descA, modeA,
                               T[beta],  B, descB, modeB,
                               T[gamma], C, descC, modeC,
                                         D, descD, modeD,
                               opAB, opABC, typeCompute, stream)
    return D
end

function elementwiseBinary!(
    alpha::Number, A::CuArray, Ainds::ModeType, opA::cutensorOperator_t,
    gamma::Number, C::CuArray{T}, Cinds::ModeType, opC::cutensorOperator_t,
    D::CuArray{T}, Dinds::ModeType, opAC::cutensorOperator_t;
    stream::CuStream=CuDefaultStream()) where {T}

    !is_unary(opA)    && throw(ArgumentError("opA must be a unary op!"))
    !is_unary(opC)    && throw(ArgumentError("opC must be a unary op!"))
    !is_binary(opAC)  && throw(ArgumentError("opAC must be a binary op!"))
    descA = CuTensorDescriptor(A; op = opA)
    descC = CuTensorDescriptor(C; op = opC)
    @assert size(C) == size(D) && strides(C) == strides(D)
    descD = descC # must currently be identical
    typeCompute = cudaDataType(T)
    modeA = collect(Cint, Ainds)
    modeC = collect(Cint, Cinds)
    modeD = modeC
    cutensorElementwiseBinary(handle(),
                              T[alpha], A, descA, modeA,
                              T[gamma], C, descC, modeC,
                                        D, descD, modeD,
                              opAC, typeCompute, stream)
    return D
end

function elementwiseBinary!(
    alpha::Number, A::Array, Ainds::ModeType, opA::cutensorOperator_t,
    gamma::Number, C::Array{T}, Cinds::ModeType, opC::cutensorOperator_t,
    D::Array{T}, Dinds::ModeType, opAC::cutensorOperator_t;
    stream::CuStream=CuDefaultStream()) where {T}

    !is_unary(opA)    && throw(ArgumentError("opA must be a unary op!"))
    !is_unary(opC)    && throw(ArgumentError("opC must be a unary op!"))
    !is_binary(opAC)  && throw(ArgumentError("opAC must be a binary op!"))
    descA = CuTensorDescriptor(A; op = opA)
    descC = CuTensorDescriptor(C; op = opC)
    @assert size(C) == size(D) && strides(C) == strides(D)
    descD = descC # must currently be identical
    typeCompute = cudaDataType(T)
    modeA = collect(Cint, Ainds)
    modeC = collect(Cint, Cinds)
    modeD = modeC
    cutensorElementwiseBinary(handle(),
                              T[alpha], A, descA, modeA,
                              T[gamma], C, descC, modeC,
                                        D, descD, modeD,
                              opAC, typeCompute, stream)
    return D
end

function elementwiseBinary!(
    alpha::Number, A::CuTensor, opA::cutensorOperator_t, gamma::Number, C::CuTensor{T}, opC::cutensorOperator_t, D::CuTensor{T}, opAC::cutensorOperator_t; stream::CuStream=CuDefaultStream()) where {T}

    !is_unary(opA)    && throw(ArgumentError("opA must be a unary op!"))
    !is_unary(opC)    && throw(ArgumentError("opC must be a unary op!"))
    !is_binary(opAC)  && throw(ArgumentError("opAC must be a binary op!"))
    descA = CuTensorDescriptor(A; op = opA)
    descC = CuTensorDescriptor(C; op = opC)
    @assert size(C) == size(D) && strides(C) == strides(D)
    descD = descC # must currently be identical
    typeCompute = cudaDataType(T)
    cutensorElementwiseBinary(handle(),
                              T[alpha], A.data, descA, A.inds,
                              T[gamma], C.data, descC, C.inds,
                                        D.data, descD, C.inds,
                              opAC, typeCompute, stream)
    return D
end

function permutation!(alpha::Number, A::CuArray, Ainds::ModeType,
                      B::CuArray, Binds::ModeType; stream::CuStream=CuDefaultStream())
    #!is_unary(opPsi)    && throw(ArgumentError("opPsi must be a unary op!"))
    descA = CuTensorDescriptor(A)
    descB = CuTensorDescriptor(B)
    T = eltype(B)
    typeCompute = cudaDataType(T)
    modeA = collect(Cint, Ainds)
    modeB = collect(Cint, Binds)
    cutensorPermutation(handle(), T[alpha], A, descA, modeA, B, descB, modeB, typeCompute,
                        stream)
    return B
end
function permutation!(alpha::Number, A::Array, Ainds::ModeType,
                      B::Array, Binds::ModeType; stream::CuStream=CuDefaultStream())
    #!is_unary(opPsi)    && throw(ArgumentError("opPsi must be a unary op!"))
    descA = CuTensorDescriptor(A)
    descB = CuTensorDescriptor(B)
    T = eltype(B)
    typeCompute = cudaDataType(T)
    modeA = collect(Cint, Ainds)
    modeB = collect(Cint, Binds)
    cutensorPermutation(handle(), T[alpha], A, descA, modeA, B, descB, modeB, typeCompute,
                        stream)
    return B
end

function contraction!(
    alpha::Number, A::CuArray, Ainds::ModeType, opA::cutensorOperator_t,
                   B::CuArray, Binds::ModeType, opB::cutensorOperator_t,
    beta::Number,  C::CuArray, Cinds::ModeType, opC::cutensorOperator_t,
                                                opOut::cutensorOperator_t;
    pref::cutensorWorksizePreference_t=CUTENSOR_WORKSPACE_RECOMMENDED,
    algo::cutensorAlgo_t=CUTENSOR_ALGO_DEFAULT, stream::CuStream=CuDefaultStream(),
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
    desc = Ref(cutensorContractionDescriptor_t(ntuple(i->0, Val(256))))
    cutensorInitContractionDescriptor(handle(),
                                      desc,
                   descA, modeA, alignmentRequirementA[],
                   descB, modeB, alignmentRequirementB[],
                   descC, modeC, alignmentRequirementC[],
                   descC, modeC, alignmentRequirementC[],
                   computeType)
    find = Ref(cutensorContractionFind_t(ntuple(i->0, Val(64))))
    cutensorInitContractionFind(handle(), find, algo)

    @workspace fallback=1<<27 size=@argout(
            cutensorContractionGetWorkspace(handle(), desc, find, pref,
                                            out(Ref{UInt64}(C_NULL)))
        )[] workspace->begin
            plan_ref = Ref(cutensorContractionPlan_t(ntuple(i->0, Val(640))))
            if isnothing(plan)
                cutensorInitContractionPlan(handle(), plan_ref, desc, find, sizeof(workspace))
            else
                plan_ref = Ref(plan)
            end
            cutensorContraction(handle(), plan_ref,
                                scalar_type[convert(scalar_type, alpha)], A, B,
                                scalar_type[convert(scalar_type, beta)],  C, C,
                                workspace, sizeof(workspace), stream)
        end
    return C
end

function plan_contraction(
    A::CuArray, Ainds::ModeType, opA::cutensorOperator_t,
    B::CuArray, Binds::ModeType, opB::cutensorOperator_t,
    C::CuArray, Cinds::ModeType, opC::cutensorOperator_t,
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
    desc = Ref(cutensorContractionDescriptor_t(ntuple(i->0, Val(256))))
    cutensorInitContractionDescriptor(handle(),
                                      desc,
                   descA, modeA, alignmentRequirementA[],
                   descB, modeB, alignmentRequirementB[],
                   descC, modeC, alignmentRequirementC[],
                   descC, modeC, alignmentRequirementC[],
                   computeType)

    find = Ref(cutensorContractionFind_t(ntuple(i->0, Val(64))))
    cutensorInitContractionFind(handle(), find, algo)
    plan = Ref(cutensorContractionPlan_t(ntuple(i->0, Val(640))))
    workspace_size = Ref{UInt64}(C_NULL)
    cutensorContractionGetWorkspace(handle(), desc, find, pref, workspace_size)
    cutensorInitContractionPlan(handle(), plan, desc, find, workspace_size[])
    return plan[]
end

function reduction!(
    alpha::Number, A::CuArray, Ainds::ModeType, opA::cutensorOperator_t,
    beta::Number,  C::CuArray, Cinds::ModeType, opC::cutensorOperator_t,
    opReduce::cutensorOperator_t; stream::CuStream=CuDefaultStream())

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

    @workspace fallback=1<<13 size=@argout(
            cutensorReductionGetWorkspace(handle(),
                A, descA, modeA,
                C, descC, modeC,
                C, descC, modeC,
                opReduce, typeCompute,
                out(Ref{UInt64}(C_NULL)))
        )[] workspace->begin
            cutensorReduction(handle(),
                T[alpha], A, descA, modeA,
                T[beta],  C, descC, modeC,
                        C, descC, modeC,
                opReduce, typeCompute,
                workspace, sizeof(workspace), stream)
        end

    return C
end

function cutensorComputeType(T::DataType)
    if T == Float32
        return CUTENSOR_R_MIN_32F
    elseif T == ComplexF32
        return CUTENSOR_C_MIN_32F
    elseif T == Float16
        return CUTENSOR_R_MIN_16F
    elseif T == ComplexF16
        return CUTENSOR_C_MIN_16F
    elseif T == Float64
        return CUTENSOR_R_MIN_64F
    elseif T == ComplexF64
        return CUTENSOR_C_MIN_64F
    elseif T == Int8
        return CUTENSOR_R_MIN_8I
    elseif T == Int32
        return CUTENSOR_R_MIN_32I
    elseif T == UInt8
        return CUTENSOR_R_MIN_8U
    elseif T == UInt32
        return CUTENSOR_R_MIN_32U
    else
        throw(ArgumentError("cutensorComputeType equivalent for input type $T does not exist!"))
    end
end
