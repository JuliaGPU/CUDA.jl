# low-level wrappers of the CUTENSOR library

using CUDAapi: cudaDataType
using CUDAdrv: CuStream_t, CuPtr, PtrOrCuPtr, CU_NULL

cutensorGetErrorString(status) = ccall((:cutensorGetErrorString,libcutensor), Cstring,
                                       (cutensorStatus_t,), status)

function cutensorCreate()
  handle = Ref{cutensorHandle_t}()
  @check ccall((:cutensorCreate, libcutensor), cutensorStatus_t,
               (Ptr{cutensorHandle_t},), handle)
  handle[]
end

function cutensorDestroy(handle)
  @check ccall((:cutensorDestroy, libcutensor), cutensorStatus_t,
               (cutensorHandle_t,), handle)
end

function cutensorCreateTensorDescriptor(numModes, extent, stride, dataType, unaryOp,
                                        vectorWidth, vectorModeIndex)
  desc = Ref{cutensorTensorDescriptor_t}(C_NULL)
  @check ccall((:cutensorCreateTensorDescriptor, libcutensor), cutensorStatus_t,
               (Ref{cutensorTensorDescriptor_t}, Cuint, Ptr{Int64}, Ptr{Int64},
                cudaDataType, cutensorOperator_t, Cuint, Cuint),
               desc, numModes, extent, stride,
               dataType, unaryOp, vectorWidth, vectorModeIndex)
  return desc[]
end

function cutensorDestroyTensorDescriptor(desc::cutensorTensorDescriptor_t)
  @check ccall((:cutensorDestroyTensorDescriptor, libcutensor), cutensorStatus_t,
               (cutensorTensorDescriptor_t,), desc)
end

function cutensorElementwiseTrinary(handle,
                                    alpha, A, descA, modeA,
                                    beta, B, descB, modeB,
                                    gamma, C, descC, modeC,
                                           D, descD, modeD,
                                    opAB, opABC, typeCompute, stream)
  @check ccall((:cutensorElementwiseTrinary,libcutensor), cutensorStatus_t,
               (cutensorHandle_t, Ptr{Cvoid}, CuPtr{Cvoid}, cutensorTensorDescriptor_t,
                Ptr{Cint}, Ptr{Cvoid}, CuPtr{Cvoid}, cutensorTensorDescriptor_t,
                Ptr{Cint}, Ptr{Cvoid}, CuPtr{Cvoid}, cutensorTensorDescriptor_t,
                Ptr{Cint}, CuPtr{Cvoid}, cutensorTensorDescriptor_t, Ptr{Cint},
                cutensorOperator_t, cutensorOperator_t, cudaDataType, CuStream_t),
               handle, alpha, A, descA, modeA, beta, B, descB, modeB, gamma, C, descC,
               modeC, D, descD, modeD, opAB, opABC, typeCompute, stream)
end

function cutensorElementwiseBinary(handle,
                                   alpha, A, descA, modeA,
                                   gamma, C, descC, modeC,
                                          D, descD, modeD,
                                   opAC, typeCompute, stream)
  @check ccall((:cutensorElementwiseBinary,libcutensor), cutensorStatus_t,
               (cutensorHandle_t, Ptr{Cvoid}, CuPtr{Cvoid}, cutensorTensorDescriptor_t,
                Ptr{Cint}, Ptr{Cvoid}, CuPtr{Cvoid}, cutensorTensorDescriptor_t,
                Ptr{Cint}, CuPtr{Cvoid}, cutensorTensorDescriptor_t, Ptr{Cint},
                cutensorOperator_t, cudaDataType, CuStream_t),
               handle, alpha, A, descA, modeA, gamma, C, descC, modeC, D, descD, modeD,
               opAC, typeCompute, stream)
end

function cutensorPermutation(handle,
                             alpha, A, descA, modeA,
                                    B, descB, modeB,
                             typeCompute, stream)
  @check ccall((:cutensorPermutation,libcutensor), cutensorStatus_t,
               (cutensorHandle_t, Ptr{Cvoid}, CuPtr{Cvoid}, cutensorTensorDescriptor_t,
                Ptr{Cint}, CuPtr{Cvoid}, cutensorTensorDescriptor_t, Ptr{Cint},
                cudaDataType, CuStream_t),
               handle, alpha, A, descA, modeA, B, descB, modeB, typeCompute, stream)
end

function cutensorContraction(handle,
                             alpha, A, descA, modeA,
                                    B, descB, modeB,
                             beta, C, descC, modeC,
                                   D, descD, modeD,
                             opOut, typeCompute, algo, workspace, workspaceSize, stream)
  @check ccall((:cutensorContraction,libcutensor), cutensorStatus_t,
               (cutensorHandle_t, Ptr{Cvoid}, CuPtr{Cvoid},
                cutensorTensorDescriptor_t, Ptr{Cint},
                CuPtr{Cvoid}, cutensorTensorDescriptor_t, Ptr{Cint},
                Ptr{Cvoid}, CuPtr{Cvoid}, cutensorTensorDescriptor_t, Ptr{Cint},
                CuPtr{Cvoid}, cutensorTensorDescriptor_t, Ptr{Cint},
                cutensorOperator_t, cudaDataType, cutensorAlgo_t, CuPtr{Cvoid},
                UInt64, CuStream_t),
               handle, alpha, A, descA, modeA, B, descB, modeB, beta, C, descC,
               modeC, D, descD, modeD, opOut, typeCompute, algo, workspace, workspaceSize,
               stream)
end

function cutensorContractionGetWorkspace(handle,
                                         A, descA, modeA,
                                         B, descB, modeB,
                                         C, descC, modeC,
                                         D, descD, modeD,
                                         opOut, typeCompute, algo, pref, workspaceSize)
  @check ccall((:cutensorContractionGetWorkspace,libcutensor), cutensorStatus_t,
               (cutensorHandle_t, CuPtr{Cvoid}, cutensorTensorDescriptor_t, Ptr{Cint},
                CuPtr{Cvoid}, cutensorTensorDescriptor_t, Ptr{Cint},
                CuPtr{Cvoid}, cutensorTensorDescriptor_t, Ptr{Cint},
                CuPtr{Cvoid}, cutensorTensorDescriptor_t, Ptr{Cint},
                cutensorOperator_t, cudaDataType, cutensorAlgo_t,
                cutensorWorksizePreference_t, Ptr{UInt64}),
               handle, A, descA, modeA, B, descB, modeB, C, descC, modeC,
               D, descD, modeD, opOut, typeCompute, algo, pref, workspaceSize)
end

function cutensorContractionMaxAlgos()
  max_algos = Ref{Cint}()
  @check ccall((:cutensorContractionMaxAlgos,libcutensor), cutensorStatus_t,
               (Ptr{Cint},), max_algos)
  return max_algos
end

function cutensorReduction(handle,
                            alpha, A, descA, modeA,
                            beta, C, descC, modeC,
                                D, descD, modeD,
                            opReduce, typeCompute, workspace, workspaceSize, stream)
  @check ccall((:cutensorReduction,libcutensor), cutensorStatus_t,
               (cutensorHandle_t,
                Ptr{Cvoid}, CuPtr{Cvoid}, cutensorTensorDescriptor_t, Ptr{Cint},
                Ptr{Cvoid}, CuPtr{Cvoid}, cutensorTensorDescriptor_t, Ptr{Cint},
                CuPtr{Cvoid}, cutensorTensorDescriptor_t, Ptr{Cint},
                cutensorOperator_t, cudaDataType, CuPtr{Cvoid}, UInt64, CuStream_t),
               handle, alpha, A, descA, modeA, beta, C, descC, modeC, D, descD, modeD,
               opReduce, typeCompute, workspace, workspaceSize, stream)
end

function cutensorReductionGetWorkspace(handle,
                            alpha, A, descA, modeA,
                            beta, C, descC, modeC,
                                D, descD, modeD,
                            opReduce, typeCompute, workspaceSize)
    @check ccall((:cutensorReductionGetWorkspace,libcutensor), cutensorStatus_t,
                 (cutensorHandle_t,
                  Ptr{Cvoid}, CuPtr{Cvoid}, cutensorTensorDescriptor_t, Ptr{Cint},
                  Ptr{Cvoid}, CuPtr{Cvoid}, cutensorTensorDescriptor_t, Ptr{Cint},
                  CuPtr{Cvoid}, cutensorTensorDescriptor_t, Ptr{Cint},
                  cutensorOperator_t, cudaDataType, Ptr{UInt64}),
                 handle, alpha, A, descA, modeA, beta, C, descC, modeC, D, descD, modeD,
                 opReduce, typeCompute, workspaceSize)
end
