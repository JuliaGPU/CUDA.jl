using CUDAapi: cudaDataType
using CUDAdrv: CuDefaultStream, CuStream

const CharUnion = Union{Char, Integer}

is_unary(op::cutensorOperator_t) = (op ∈ [CUTENSOR_OP_IDENTITY, CUTENSOR_OP_SQRT, CUTENSOR_OP_RELU, CUTENSOR_OP_CONJ, CUTENSOR_OP_RCP])
is_binary(op::cutensorOperator_t) = (op ∈ [CUTENSOR_OP_ADD, CUTENSOR_OP_MUL, CUTENSOR_OP_MAX, CUTENSOR_OP_MIN])

function cutensorDescriptor(A, op::cutensorOperator_t)
    dimsA = collect(size(A))
    stridA = collect(strides(A))
    return cutensorCreateTensorDescriptor(Cint(ndims(A)), dimsA, stridA, cudaDataType(eltype(A)), op, Cint(1), Cint(0))
end
#=
function cutensorDescriptor(A::CuTensor, op::cutensorOperator_t)
    dimsA = collect(size(A))
    stridA = collect(strides(A))
    return cutensorCreateTensorDescriptor(Cint(ndims(A)), dimsA, stridA, cudaDataType(eltype(A)), op, Cint(1), Cint(0))
end
=#
function elementwiseTrinary!(alpha::Number, A::CuArray, Ainds::Vector{<:CharUnion}, opA::cutensorOperator_t,
                             beta::Number, B::CuArray, Binds::Vector{<:CharUnion}, opB::cutensorOperator_t,
                             gamma::Number, C::CuArray, Cinds::Vector{<:CharUnion}, opC::cutensorOperator_t,
                             D::CuArray, Dinds::Vector{<:CharUnion},
                             opAB::cutensorOperator_t, opABC::cutensorOperator_t; stream::CuStream=CuDefaultStream())
    !is_unary(opA)    && throw(ArgumentError("opA must be a unary op!"))
    !is_unary(opB)    && throw(ArgumentError("opB must be a unary op!"))
    !is_unary(opC)    && throw(ArgumentError("opC must be a unary op!"))
    !is_binary(opAB)  && throw(ArgumentError("opAB must be a binary op!"))
    !is_binary(opABC) && throw(ArgumentError("opABC must be a binary op!"))
    descA = cutensorDescriptor(A, opA)
    descB = cutensorDescriptor(B, opB)
    descC = cutensorDescriptor(C, opC)
    descD = descC # must currently be identical
    typeCompute = cudaDataType(eltype(C))
    modeA = Vector{Cwchar_t}(Ainds)
    modeB = Vector{Cwchar_t}(Binds)
    modeC = Vector{Cwchar_t}(Cinds)
    modeD = modeC
    cutensorElementwiseTrinary(handle(), [alpha], A, descA, modeA, [beta], B, descB, modeB,
                               [gamma], C, descC, modeC, D, descD, modeD, opAB, opABC,
                               typeCompute, stream)
    return D
end

function elementwiseTrinary!(alpha::Number, A::Array, Ainds::Vector{<:CharUnion}, opA::cutensorOperator_t,
                             beta::Number, B::Array, Binds::Vector{<:CharUnion}, opB::cutensorOperator_t,
                             gamma::Number, C::Array, Cinds::Vector{<:CharUnion}, opC::cutensorOperator_t,
                             D::Array, Dinds::Vector{<:CharUnion},
                             opAB::cutensorOperator_t, opABC::cutensorOperator_t; stream::CuStream=CuDefaultStream())
    !is_unary(opA)    && throw(ArgumentError("opA must be a unary op!"))
    !is_unary(opB)    && throw(ArgumentError("opB must be a unary op!"))
    !is_unary(opC)    && throw(ArgumentError("opC must be a unary op!"))
    !is_binary(opAB)  && throw(ArgumentError("opAB must be a binary op!"))
    !is_binary(opABC) && throw(ArgumentError("opABC must be a binary op!"))
    descA = cutensorDescriptor(A, opA)
    descB = cutensorDescriptor(B, opB)
    descC = cutensorDescriptor(C, opC)
    descD = descC # must currently be identical
    typeCompute = cudaDataType(eltype(C))
    modeA = Vector{Cwchar_t}(Ainds)
    modeB = Vector{Cwchar_t}(Binds)
    modeC = Vector{Cwchar_t}(Cinds)
    modeD = modeC
    cutensorElementwiseTrinary(handle(), [alpha], A, descA, modeA, [beta], B, descB, modeB,
                               [gamma], C, descC, modeC, D, descD, modeD, opAB, opABC,
                               typeCompute, stream)
    return D
end


function elementwiseBinary!(alpha::Number, A::CuArray, Ainds::Vector{<:CharUnion}, opA::cutensorOperator_t,
                            gamma::Number, C::CuArray, Cinds::Vector{<:CharUnion}, opC::cutensorOperator_t,
                            D::CuArray, Dinds::Vector{<:CharUnion}, opAC::cutensorOperator_t; stream::CuStream=CuDefaultStream())
    !is_unary(opA)    && throw(ArgumentError("opA must be a unary op!"))
    !is_unary(opC)    && throw(ArgumentError("opC must be a unary op!"))
    !is_binary(opAC)  && throw(ArgumentError("opAC must be a binary op!"))
    descA = cutensorDescriptor(A, opA)
    descC = cutensorDescriptor(C, opC)
    descD = descC # must currently be identical
    typeCompute = cudaDataType(eltype(D))
    modeA = Vector{Cwchar_t}(Ainds)
    modeC = Vector{Cwchar_t}(Cinds)
    modeD = modeC
    cutensorElementwiseBinary(handle(), [alpha], A, descA, modeA, [gamma], C, descC,
                              modeC, D, descD, modeD, opAC, typeCompute, stream)
    return D
end
function elementwiseBinary!(alpha::Number, A::Array, Ainds::Vector{<:CharUnion}, opA::cutensorOperator_t,
                            gamma::Number, C::Array, Cinds::Vector{<:CharUnion}, opC::cutensorOperator_t,
                            D::Array, Dinds::Vector{<:CharUnion}, opAC::cutensorOperator_t; stream::CuStream=CuDefaultStream())
    !is_unary(opA)    && throw(ArgumentError("opA must be a unary op!"))
    !is_unary(opC)    && throw(ArgumentError("opC must be a unary op!"))
    !is_binary(opAC)  && throw(ArgumentError("opAC must be a binary op!"))
    descA = cutensorDescriptor(A, opA)
    descC = cutensorDescriptor(C, opC)
    descD = descC # must currently be identical
    typeCompute = cudaDataType(eltype(D))
    modeA = Vector{Cwchar_t}(Ainds)
    modeC = Vector{Cwchar_t}(Cinds)
    modeD = modeC
    cutensorElementwiseBinary(handle(), [alpha], A, descA, modeA, [gamma], C, descC,
                              modeC, D, descD, modeD, opAC, typeCompute, stream)
    return D
end
function elementwiseBinary!(alpha::Number, A::CuTensor, opA::cutensorOperator_t,
                            gamma::Number, C::CuTensor, opC::cutensorOperator_t,
                            D::CuTensor, opAC::cutensorOperator_t; stream::CuStream=CuDefaultStream())
    !is_unary(opA)    && throw(ArgumentError("opA must be a unary op!"))
    !is_unary(opC)    && throw(ArgumentError("opC must be a unary op!"))
    !is_binary(opAC)  && throw(ArgumentError("opAC must be a binary op!"))
    descA = cutensorDescriptor(A, opA)
    descC = cutensorDescriptor(C, opC)
    typeCompute = cudaDataType(eltype(D))
    cutensorElementwiseBinary(handle(), [alpha], A.data, descA, A.inds, [gamma], C.data,
                              descC, C.inds, D.data, descC, C.inds, opAC, typeCompute,
                              stream)
    return D
end

function permutation!(alpha::Number, A::CuArray, Ainds::Vector{<:CharUnion},
                      B::CuArray, Binds::Vector{<:CharUnion}; stream::CuStream=CuDefaultStream())
    #!is_unary(opPsi)    && throw(ArgumentError("opPsi must be a unary op!"))
    descA = cutensorDescriptor(A, CUTENSOR_OP_IDENTITY)
    descB = cutensorDescriptor(B, CUTENSOR_OP_IDENTITY)
    typeCompute = cudaDataType(eltype(B))
    modeA = Vector{Cwchar_t}(Ainds)
    modeB = Vector{Cwchar_t}(Binds)
    cutensorPermutation(handle(), [alpha], A, descA, modeA, B, descB, modeB, typeCompute,
                        stream)
    return B
end

function permutation!(alpha::Number, A::Array, Ainds::Vector{<:CharUnion},
                      B::Array, Binds::Vector{<:CharUnion}; stream::CuStream=CuDefaultStream())
    #!is_unary(opPsi)    && throw(ArgumentError("opPsi must be a unary op!"))
    descA = cutensorDescriptor(A, CUTENSOR_OP_IDENTITY)
    descB = cutensorDescriptor(B, CUTENSOR_OP_IDENTITY)
    typeCompute = cudaDataType(eltype(B))
    modeA = Vector{Cwchar_t}(Ainds)
    modeB = Vector{Cwchar_t}(Binds)
    cutensorPermutation(handle(), [alpha], A, descA, modeA, B, descB, modeB, typeCompute,
                        stream)
    return B
end

function contraction!(alpha::Number, A::CuArray, Ainds::Vector{<:CharUnion}, opA::cutensorOperator_t,
                      B::CuArray, Binds::Vector{<:CharUnion}, opB::cutensorOperator_t,
                      beta::Number, C::CuArray, Cinds::Vector{<:CharUnion}, opC::cutensorOperator_t,
                      opOut::cutensorOperator_t; pref::cutensorWorksizePreference_t=CUTENSOR_WORKSPACE_RECOMMENDED,
                      algo::cutensorAlgo_t=CUTENSOR_ALGO_DEFAULT, stream::CuStream=CuDefaultStream())
    !is_unary(opA)    && throw(ArgumentError("opA must be a unary op!"))
    !is_unary(opB)    && throw(ArgumentError("opB must be a unary op!"))
    !is_unary(opC)    && throw(ArgumentError("opC must be a unary op!"))
    !is_unary(opOut)  && throw(ArgumentError("opOut must be a unary op!"))
    descA = cutensorDescriptor(A, opA)
    descB = cutensorDescriptor(B, opB)
    descC = cutensorDescriptor(C, opC)
    # for now, descD must be identical to descC
    typeCompute = cudaDataType(eltype(C))
    descD = descC
    modeA = Vector{Cwchar_t}(Ainds)
    modeB = Vector{Cwchar_t}(Binds)
    modeC = Vector{Cwchar_t}(Cinds)
    workspaceSize = Ref{UInt64}(C_NULL)
    cutensorContractionGetWorkspace(handle(), A, descA, modeA, B, descB, modeB, C, descC,
                                    modeC, C, descC, modeC, opOut, typeCompute, algo, pref,
                                    workspaceSize)
    workspace = cuzeros(eltype(C), workspaceSize[])
    cutensorContraction(handle(), [alpha], A, descA, modeA, B, descB, modeB, [beta], C,
                        descC, modeC, C, descC, modeC, opOut, typeCompute, algo, workspace,
                        workspaceSize[], stream)
    return C
end

function contraction!(alpha::Number, A::Array, Ainds::Vector{<:CharUnion}, opA::cutensorOperator_t,
                      B::Array, Binds::Vector{<:CharUnion}, opB::cutensorOperator_t,
                      beta::Number, C::Array, Cinds::Vector{<:CharUnion}, opC::cutensorOperator_t,
                      opOut::cutensorOperator_t; pref::cutensorWorksizePreference_t=CUTENSOR_WORKSPACE_RECOMMENDED,
                      algo::cutensorAlgo_t=CUTENSOR_ALGO_DEFAULT, stream::CuStream=CuDefaultStream())
    !is_unary(opA)    && throw(ArgumentError("opA must be a unary op!"))
    !is_unary(opB)    && throw(ArgumentError("opB must be a unary op!"))
    !is_unary(opC)    && throw(ArgumentError("opC must be a unary op!"))
    !is_unary(opOut)  && throw(ArgumentError("opOut must be a unary op!"))
    descA = cutensorDescriptor(A, opA)
    descB = cutensorDescriptor(B, opB)
    descC = cutensorDescriptor(C, opC)
    # for now, descD must be identical to descC
    typeCompute = cudaDataType(eltype(C))
    descD = descC
    modeA = Vector{Cwchar_t}(Ainds)
    modeB = Vector{Cwchar_t}(Binds)
    modeC = Vector{Cwchar_t}(Cinds)
    cutensorContraction(handle(), [alpha], A, descA, modeA, B, descB, modeB, [beta], C,
                        descC, modeC, C, descC, modeC, opOut, typeCompute, algo, CU_NULL, 0,
                        stream)
    return C
end

function contraction!(alpha::Number, A::CuTensor, opA::cutensorOperator_t, B::CuTensor, opB::cutensorOperator_t,
                      beta::Number, C::CuTensor, opC::cutensorOperator_t, opOut::cutensorOperator_t;
                      pref::cutensorWorksizePreference_t=CUTENSOR_WORKSPACE_RECOMMENDED,
                      algo::cutensorAlgo_t=CUTENSOR_ALGO_DEFAULT, stream::CuStream=CuDefaultStream())
    !is_unary(opA)    && throw(ArgumentError("opA must be a unary op!"))
    !is_unary(opB)    && throw(ArgumentError("opB must be a unary op!"))
    !is_unary(opC)    && throw(ArgumentError("opC must be a unary op!"))
    !is_unary(opOut)  && throw(ArgumentError("opOut must be a unary op!"))
    descA = cutensorDescriptor(A, opA)
    descB = cutensorDescriptor(B, opB)
    descC = cutensorDescriptor(C, opC)
    # for now, descD must be identical to descC
    typeCompute = cudaDataType(eltype(C))
    descD = descC
    workspaceSize = Ref{UInt64}(C_NULL)
    cutensorContractionGetWorkspace(handle(), A.data, descA, A.inds, B.data, descB, B.inds,
                                    C.data, descC, C.inds, C.data, descC, C.inds, opOut,
                                    typeCompute, algo, pref, workspaceSize)
    workspace = cuzeros(eltype(C), workspaceSize[])
    cutensorContraction(handle(), [alpha], A.data, descA, A.inds, B.data, descB, B.inds,
                        [beta], C.data, descC, C.inds, C.data, descC, C.inds, opOut,
                        typeCompute, algo, workspace, workspaceSize[], stream)
    return C
end
