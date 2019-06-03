# low-level wrappers of the CUTENSOR library

using CUDAapi: cudaDataType
using CUDAdrv: CuStream_t, CuPtr, PtrOrCuPtr, CU_NULL

cutensorGetErrorString(status) = ccall((:cutensorGetErrorString,libcutensor), Ptr{UInt8},
                                       (cutensorStatus_t,), status)

function cutensorCreate()
  handle = Ref{cutensorHandle_t}()
  @check ccall((:cutensorCreate, libcutensor), cutensorStatus_t,
               (Ptr{cutensorHandle_t},), handle)
  handle[]
end

function cutensorDestroy(handle)
  @check ccall((:cutensorDestroy, libcutensor), cutensorStatus_t, (cutensorHandle_t,), handle)
end

function cutensorCreateTensorDescriptor(numModes::Cint,
                                        extent::Vector{Int64},
                                        stride::Vector{Int64},
                                        T::cudaDataType,
                                        unaryOp::cutensorOperator_t,
                                        vectorWidth::Cint,
                                        vectorModeIndex::Cint)
  desc = Ref{cutensorTensorDescriptor_t}(C_NULL)
  @check ccall((:cutensorCreateTensorDescriptor, libcutensor), cutensorStatus_t,
               (Ref{cutensorTensorDescriptor_t}, Cint, Ptr{Int64}, Ptr{Int64},
                cudaDataType, cutensorOperator_t, Cint, Cint),
               desc, numModes, extent, stride, T, unaryOp, vectorWidth, vectorModeIndex)
  return desc[]
end

function cutensorDestroyTensorDescriptor(desc::cutensorTensorDescriptor_t)
  @check ccall((:cutensorDestroyTensorDescriptor, libcutensor), cutensorStatus_t,
               (cutensorTensorDescriptor_t,), desc)
end

for (AT, PT) in ((Array, Ptr{Cvoid}), (CuArray, CuPtr{Cvoid}))
    @eval function cutensorElementwiseTrinary(handle,
                                        alpha, A::$AT, descA, modeA,
                                        beta, B::$AT, descB, modeB,
                                        gamma, C::$AT, descC, modeC,
                                               D::$AT, descD, modeD,
                                        opAB, opABC, typeCompute, stream)
      @check ccall((:cutensorElementwiseTrinary,libcutensor), cutensorStatus_t,
                   (cutensorHandle_t, Ptr{Cvoid}, $PT, cutensorTensorDescriptor_t,
                    Ptr{Cint}, Ptr{Cvoid}, $PT, cutensorTensorDescriptor_t,
                    Ptr{Cint}, Ptr{Cvoid}, $PT, cutensorTensorDescriptor_t,
                    Ptr{Cint}, $PT, cutensorTensorDescriptor_t, Ptr{Cint},
                    cutensorOperator_t, cutensorOperator_t, cudaDataType, CuStream_t),
                   handle, alpha, A, descA, modeA, beta, B, descB, modeB, gamma, C, descC,
                   modeC, D, descD, modeD, opAB, opABC, typeCompute, stream)
    end

    @eval function cutensorElementwiseBinary(handle,
                                       alpha, A::$AT, descA, modeA,
                                       gamma, C::$AT, descC, modeC,
                                              D::$AT, descD, modeD,
                                       opAC, typeCompute, stream)
      @check ccall((:cutensorElementwiseBinary,libcutensor), cutensorStatus_t,
                   (cutensorHandle_t, Ptr{Cvoid}, $PT, cutensorTensorDescriptor_t,
                    Ptr{Cint}, Ptr{Cvoid}, $PT, cutensorTensorDescriptor_t,
                    Ptr{Cint}, $PT, cutensorTensorDescriptor_t, Ptr{Cint},
                    cutensorOperator_t, cudaDataType, CuStream_t),
                   handle, alpha, A, descA, modeA, gamma, C, descC, modeC, D, descD, modeD,
                   opAC, typeCompute, stream)
    end

    @eval function cutensorPermutation(handle,
                                 alpha, A::$AT, descA, modeA,
                                        B::$AT, descB, modeB,
                                 typeCompute, stream)
      @check ccall((:cutensorPermutation,libcutensor), cutensorStatus_t,
                   (cutensorHandle_t, Ptr{Cvoid}, $PT, cutensorTensorDescriptor_t,
                    Ptr{Cint}, $PT, cutensorTensorDescriptor_t, Ptr{Cint},
                    cudaDataType, CuStream_t),
                   handle, alpha, A, descA, modeA, B, descB, modeB, typeCompute, stream)
    end

    @eval function cutensorContraction(handle,
                                 alpha, A::$AT, descA, modeA,
                                        B::$AT, descB, modeB,
                                 beta, C::$AT, descC, modeC,
                                       D::$AT, descD, modeD,
                                 opOut, typeCompute, algo, workspace, workspaceSize, stream)
      @check ccall((:cutensorContraction,libcutensor), cutensorStatus_t,
                   (cutensorHandle_t, Ptr{Cvoid}, $PT, cutensorTensorDescriptor_t, Ptr{Cint},
                    $PT, cutensorTensorDescriptor_t, Ptr{Cint},
                    Ptr{Cvoid}, $PT, cutensorTensorDescriptor_t, Ptr{Cint},
                    $PT, cutensorTensorDescriptor_t, Ptr{Cint},
                    cutensorOperator_t, cudaDataType, cutensorAlgo_t, CuPtr{Cvoid},
                    UInt64, CuStream_t),
                   handle, alpha, A, descA, modeA, B, descB, modeB, beta, C, descC,
                   modeC, D, descD, modeD, opOut, typeCompute, algo, workspace, workspaceSize,
                   stream)
    end

    @eval function cutensorContractionGetWorkspace(handle,
                                             A::$AT, descA, modeA,
                                             B::$AT, descB, modeB,
                                             C::$AT, descC, modeC,
                                             D::$AT, descD, modeD,
                                             opOut, typeCompute, algo, pref, workspaceSize)
      @check ccall((:cutensorContractionGetWorkspace,libcutensor), cutensorStatus_t,
                   (cutensorHandle_t, $PT, cutensorTensorDescriptor_t, Ptr{Cint},
                    $PT, cutensorTensorDescriptor_t, Ptr{Cint},
                    $PT, cutensorTensorDescriptor_t, Ptr{Cint},
                    $PT, cutensorTensorDescriptor_t, Ptr{Cint},
                    cutensorOperator_t, cudaDataType, cutensorAlgo_t, cutensorWorksizePreference_t,
                    Ptr{UInt64}),
                   handle, A, descA, modeA, B, descB, modeB, C, descC, modeC,
                   D, descD, modeD, opOut, typeCompute, algo, pref, workspaceSize)
    end
end

function cutensorContractionMaxAlgos()
  max_algos = Ref{Cint}()
  @check ccall((:cutensorContractionMaxAlgos,libcutensor), cutensorStatus_t,
               (Ptr{Cint},), max_algos)
  return max_algos
end
