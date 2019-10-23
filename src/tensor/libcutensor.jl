# Julia wrapper for header: types.h
# Automatically generated using Clang.jl

# Julia wrapper for header: cutensor.h
# Automatically generated using Clang.jl


function cutensorCreate(handle)
    @check ccall((:cutensorCreate, @libcutensor), cutensorStatus_t,
                 (Ptr{cutensorHandle_t},),
                 handle)
end

function cutensorDestroy(handle)
    @check ccall((:cutensorDestroy, @libcutensor), cutensorStatus_t,
                 (cutensorHandle_t,),
                 handle)
end

function cutensorCreateTensorDescriptor(desc, numModes, extent, stride, dataType, unaryOp,
                                        vectorWidth, vectorModeIndex)
    @check ccall((:cutensorCreateTensorDescriptor, @libcutensor), cutensorStatus_t,
                 (Ptr{cutensorTensorDescriptor_t}, UInt32, Ptr{Int64}, Ptr{Int64},
                  cudaDataType_t, cutensorOperator_t, UInt32, UInt32),
                 desc, numModes, extent, stride, dataType, unaryOp, vectorWidth,
                 vectorModeIndex)
end

function cutensorDestroyTensorDescriptor(desc)
    @check ccall((:cutensorDestroyTensorDescriptor, @libcutensor), cutensorStatus_t,
                 (cutensorTensorDescriptor_t,),
                 desc)
end

function cutensorElementwiseTrinary(handle, alpha, A, descA, modeA, beta, B, descB, modeB,
                                    gamma, C, descC, modeC, D, descD, modeD, opAB, opABC,
                                    typeCompute, stream)
    @check ccall((:cutensorElementwiseTrinary, @libcutensor), cutensorStatus_t,
                 (cutensorHandle_t, Ptr{Cvoid}, CuPtr{Cvoid}, cutensorTensorDescriptor_t,
                  Ptr{Int32}, Ptr{Cvoid}, CuPtr{Cvoid}, cutensorTensorDescriptor_t,
                  Ptr{Int32}, Ptr{Cvoid}, CuPtr{Cvoid}, cutensorTensorDescriptor_t,
                  Ptr{Int32}, CuPtr{Cvoid}, cutensorTensorDescriptor_t, Ptr{Int32},
                  cutensorOperator_t, cutensorOperator_t, cudaDataType_t, CUstream),
                 handle, alpha, A, descA, modeA, beta, B, descB, modeB, gamma, C, descC,
                 modeC, D, descD, modeD, opAB, opABC, typeCompute, stream)
end

function cutensorElementwiseBinary(handle, alpha, A, descA, modeA, gamma, C, descC, modeC,
                                   D, descD, modeD, opAC, typeCompute, stream)
    @check ccall((:cutensorElementwiseBinary, @libcutensor), cutensorStatus_t,
                 (cutensorHandle_t, Ptr{Cvoid}, CuPtr{Cvoid}, cutensorTensorDescriptor_t,
                  Ptr{Int32}, Ptr{Cvoid}, CuPtr{Cvoid}, cutensorTensorDescriptor_t,
                  Ptr{Int32}, CuPtr{Cvoid}, cutensorTensorDescriptor_t, Ptr{Int32},
                  cutensorOperator_t, cudaDataType_t, CUstream),
                 handle, alpha, A, descA, modeA, gamma, C, descC, modeC, D, descD, modeD,
                 opAC, typeCompute, stream)
end

function cutensorPermutation(handle, alpha, A, descA, modeA, B, descB, modeB, typeCompute,
                             stream)
    @check ccall((:cutensorPermutation, @libcutensor), cutensorStatus_t,
                 (cutensorHandle_t, Ptr{Cvoid}, CuPtr{Cvoid}, cutensorTensorDescriptor_t,
                  Ptr{Int32}, CuPtr{Cvoid}, cutensorTensorDescriptor_t, Ptr{Int32},
                  cudaDataType_t, CUstream),
                 handle, alpha, A, descA, modeA, B, descB, modeB, typeCompute, stream)
end

function cutensorContraction(handle, alpha, A, descA, modeA, B, descB, modeB, beta, C,
                             descC, modeC, D, descD, modeD, opOut, typeCompute, algo,
                             workspace, workspaceSize, stream)
    @check ccall((:cutensorContraction, @libcutensor), cutensorStatus_t,
                 (cutensorHandle_t, Ptr{Cvoid}, CuPtr{Cvoid}, cutensorTensorDescriptor_t,
                  Ptr{Int32}, CuPtr{Cvoid}, cutensorTensorDescriptor_t, Ptr{Int32},
                  Ptr{Cvoid}, CuPtr{Cvoid}, cutensorTensorDescriptor_t, Ptr{Int32},
                  CuPtr{Cvoid}, cutensorTensorDescriptor_t, Ptr{Int32}, cutensorOperator_t,
                  cudaDataType_t, cutensorAlgo_t, CuPtr{Cvoid}, UInt64, CUstream),
                 handle, alpha, A, descA, modeA, B, descB, modeB, beta, C, descC, modeC, D,
                 descD, modeD, opOut, typeCompute, algo, workspace, workspaceSize, stream)
end

function cutensorContractionGetWorkspace(handle, A, descA, modeA, B, descB, modeB, C,
                                         descC, modeC, D, descD, modeD, opOut, typeCompute,
                                         algo, pref, workspaceSize)
    @check ccall((:cutensorContractionGetWorkspace, @libcutensor), cutensorStatus_t,
                 (cutensorHandle_t, CuPtr{Cvoid}, cutensorTensorDescriptor_t, Ptr{Int32},
                  CuPtr{Cvoid}, cutensorTensorDescriptor_t, Ptr{Int32}, CuPtr{Cvoid},
                  cutensorTensorDescriptor_t, Ptr{Int32}, CuPtr{Cvoid},
                  cutensorTensorDescriptor_t, Ptr{Int32}, cutensorOperator_t,
                  cudaDataType_t, cutensorAlgo_t, cutensorWorksizePreference_t, Ptr{UInt64}),
                 handle, A, descA, modeA, B, descB, modeB, C, descC, modeC, D, descD,
                 modeD, opOut, typeCompute, algo, pref, workspaceSize)
end

function cutensorContractionMaxAlgos(maxNumAlgos)
    @check ccall((:cutensorContractionMaxAlgos, @libcutensor), cutensorStatus_t,
                 (Ptr{Int32},),
                 maxNumAlgos)
end

function cutensorReduction(handle, alpha, A, descA, modeA, beta, C, descC, modeC, D, descD,
                           modeD, opReduce, typeCompute, workspace, workspaceSize, stream)
    @check ccall((:cutensorReduction, @libcutensor), cutensorStatus_t,
                 (cutensorHandle_t, Ptr{Cvoid}, CuPtr{Cvoid}, cutensorTensorDescriptor_t,
                  Ptr{Int32}, Ptr{Cvoid}, CuPtr{Cvoid}, cutensorTensorDescriptor_t,
                  Ptr{Int32}, CuPtr{Cvoid}, cutensorTensorDescriptor_t, Ptr{Int32},
                  cutensorOperator_t, cudaDataType_t, CuPtr{Cvoid}, UInt64, CUstream),
                 handle, alpha, A, descA, modeA, beta, C, descC, modeC, D, descD, modeD,
                 opReduce, typeCompute, workspace, workspaceSize, stream)
end

function cutensorReductionGetWorkspace(handle, A, descA_, modeA, C, descC_, modeC, D,
                                       descD_, modeD, opReduce, typeCompute, workspaceSize)
    @check ccall((:cutensorReductionGetWorkspace, @libcutensor), cutensorStatus_t,
                 (cutensorHandle_t, CuPtr{Cvoid}, cutensorTensorDescriptor_t, Ptr{Int32},
                  CuPtr{Cvoid}, cutensorTensorDescriptor_t, Ptr{Int32}, CuPtr{Cvoid},
                  cutensorTensorDescriptor_t, Ptr{Int32}, cutensorOperator_t,
                  cudaDataType_t, Ptr{UInt64}),
                 handle, A, descA_, modeA, C, descC_, modeC, D, descD_, modeD, opReduce,
                 typeCompute, workspaceSize)
end

function cutensorGetErrorString(error)
    ccall((:cutensorGetErrorString, @libcutensor), Cstring,
          (cutensorStatus_t,),
          error)
end

function cutensorGetVersion()
    ccall((:cutensorGetVersion, @libcutensor), Csize_t, ())
end

function cutensorGetCudartVersion()
    ccall((:cutensorGetCudartVersion, @libcutensor), Csize_t, ())
end
