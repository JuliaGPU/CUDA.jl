using CUDAapi: cudaDataType
using CUDAdrv: CuDefaultStream, CuStream

const ModeType = AbstractVector{<:Union{Char, Integer}}

is_unary(op::cutensorOperator_t) =
    (op ∈ (CUTENSOR_OP_IDENTITY, CUTENSOR_OP_SQRT, CUTENSOR_OP_RELU, CUTENSOR_OP_CONJ,
            CUTENSOR_OP_RCP))
is_binary(op::cutensorOperator_t) =
    (op ∈ (CUTENSOR_OP_ADD, CUTENSOR_OP_MUL, CUTENSOR_OP_MAX, CUTENSOR_OP_MIN))

mutable struct CuTensorDescriptor
    desc::cutensorTensorDescriptor_t

    function CuTensorDescriptor(a; size = size(a), strides = strides(a), eltype = eltype(a),
                                   op = CUTENSOR_OP_IDENTITY,
                                   vectorwidth = Cint(1), vectormodeindex = Cint(0))
        desc = cutensorCreateTensorDescriptor(ndims(a), collect(Int64, size),
                                                collect(Int64, strides),
                                                cudaDataType(eltype), op,
                                                vectorwidth, vectormodeindex)
        obj = new(desc)
        finalizer(destroy!, obj)
        return obj
    end
end
destroy!(obj::CuTensorDescriptor) = cutensorDestroyTensorDescriptor(obj.desc)
Base.cconvert(::Type{cutensorTensorDescriptor_t}, obj::CuTensorDescriptor) = obj.desc

function elementwiseTrinary!(
    alpha::Number, A::CuArray, Ainds::ModeType, opA::cutensorOperator_t,
    beta::Number, B::CuArray, Binds::ModeType, opB::cutensorOperator_t,
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
    cutensorElementwiseTrinary(handle(), T[alpha], A, descA, modeA,
                                T[beta], B, descB, modeB, T[gamma], C, descC, modeC,
                                D, descD, modeD, opAB, opABC, typeCompute, stream)
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
    cutensorElementwiseTrinary(handle(), T[alpha], A, descA, modeA,
                                T[beta], B, descB, modeB, T[gamma], C, descC, modeC,
                                D, descD, modeD, opAB, opABC, typeCompute, stream)
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
    cutensorElementwiseBinary(handle(), T[alpha], A, descA, modeA, T[gamma], C, descC,
                              modeC, D, descD, modeD, opAC, typeCompute, stream)
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
    cutensorElementwiseBinary(handle(), T[alpha], A, descA, modeA, T[gamma], C, descC,
                              modeC, D, descD, modeD, opAC, typeCompute, stream)
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
    cutensorElementwiseBinary(handle(), T[alpha], A.data, descA, A.inds, T[gamma], C.data,
                              descC, C.inds, D.data, descD, C.inds, opAC, typeCompute,
                              stream)
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
    beta::Number, C::CuArray, Cinds::ModeType, opC::cutensorOperator_t,
    opOut::cutensorOperator_t;
    pref::cutensorWorksizePreference_t=CUTENSOR_WORKSPACE_RECOMMENDED,
    algo::cutensorAlgo_t=CUTENSOR_ALGO_DEFAULT, stream::CuStream=CuDefaultStream())

    !is_unary(opA)    && throw(ArgumentError("opA must be a unary op!"))
    !is_unary(opB)    && throw(ArgumentError("opB must be a unary op!"))
    !is_unary(opC)    && throw(ArgumentError("opC must be a unary op!"))
    !is_unary(opOut)  && throw(ArgumentError("opOut must be a unary op!"))
    descA = CuTensorDescriptor(A; op = opA)
    descB = CuTensorDescriptor(B; op = opB)
    descC = CuTensorDescriptor(C; op = opC)
    # for now, D must be identical to C (and thus, descD must be identical to descC)
    T = eltype(C)
    typeCompute = cudaDataType(T)
    modeA = collect(Cint, Ainds)
    modeB = collect(Cint, Binds)
    modeC = collect(Cint, Cinds)
    workspaceSize = Ref{UInt64}(C_NULL)
    cutensorContractionGetWorkspace(handle(), A, descA, modeA, B, descB, modeB, C, descC,
                                    modeC, C, descC, modeC, opOut, typeCompute, algo, pref,
                                    workspaceSize)
    workspace = CuArray{T}(undef, workspaceSize[])
    cutensorContraction(handle(), T[alpha], A, descA, modeA, B, descB, modeB, T[beta], C,
                        descC, modeC, C, descC, modeC, opOut, typeCompute, algo, workspace,
                        workspaceSize[], stream)
    return C
end

function contraction!(
    alpha::Number, A::Array, Ainds::ModeType, opA::cutensorOperator_t,
    B::Array, Binds::ModeType, opB::cutensorOperator_t,
    beta::Number, C::Array, Cinds::ModeType, opC::cutensorOperator_t,
    opOut::cutensorOperator_t;
    pref::cutensorWorksizePreference_t=CUTENSOR_WORKSPACE_RECOMMENDED,
    algo::cutensorAlgo_t=CUTENSOR_ALGO_DEFAULT, stream::CuStream=CuDefaultStream())

    !is_unary(opA)    && throw(ArgumentError("opA must be a unary op!"))
    !is_unary(opB)    && throw(ArgumentError("opB must be a unary op!"))
    !is_unary(opC)    && throw(ArgumentError("opC must be a unary op!"))
    !is_unary(opOut)  && throw(ArgumentError("opOut must be a unary op!"))
    descA = CuTensorDescriptor(A; op = opA)
    descB = CuTensorDescriptor(B; op = opB)
    descC = CuTensorDescriptor(C; op = opC)
    # for now, D must be identical to C (and thus, descD must be identical to descC)
    T = eltype(C)
    typeCompute = cudaDataType(T)
    modeA = collect(Cint, Ainds)
    modeB = collect(Cint, Binds)
    modeC = collect(Cint, Cinds)
    cutensorContraction(handle(), T[alpha], A, descA, modeA, B, descB, modeB, T[beta], C,
                        descC, modeC, C, descC, modeC, opOut, typeCompute, algo, CU_NULL, 0,
                        stream)
    return C
end
function contraction!(alpha::Number,
    A::CuTensor, opA::cutensorOperator_t, B::CuTensor, opB::cutensorOperator_t,
    beta::Number, C::CuTensor, opC::cutensorOperator_t, opOut::cutensorOperator_t;
    pref::cutensorWorksizePreference_t=CUTENSOR_WORKSPACE_RECOMMENDED,
    algo::cutensorAlgo_t=CUTENSOR_ALGO_DEFAULT, stream::CuStream=CuDefaultStream())

    !is_unary(opA)    && throw(ArgumentError("opA must be a unary op!"))
    !is_unary(opB)    && throw(ArgumentError("opB must be a unary op!"))
    !is_unary(opC)    && throw(ArgumentError("opC must be a unary op!"))
    !is_unary(opOut)  && throw(ArgumentError("opOut must be a unary op!"))
    descA = CuTensorDescriptor(A; op = opA)
    descB = CuTensorDescriptor(B; op = opB)
    descC = CuTensorDescriptor(C; op = opC)
    # for now, D must be identical to C (and thus, descD must be identical to descC)
    T = eltype(C)
    typeCompute = cudaDataType(T)
    workspaceSize = Ref{UInt64}(C_NULL)
    cutensorContractionGetWorkspace(handle(), A.data, descA, A.inds, B.data, descB, B.inds,
                                    C.data, descC, C.inds, C.data, descC, C.inds, opOut,
                                    typeCompute, algo, pref, workspaceSize)
    workspace = CuArray{T}(undef, workspaceSize[])
    cutensorContraction(handle(), T[alpha], A.data, descA, A.inds, B.data, descB, B.inds,
                        T[beta], C.data, descC, C.inds, C.data, descC, C.inds, opOut,
                        typeCompute, algo, workspace, workspaceSize[], stream)
    return C
end
