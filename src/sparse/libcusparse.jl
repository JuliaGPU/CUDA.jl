# Julia wrapper for header: cusparse.h
# Automatically generated using Clang.jl


function cusparseCreate(handle)
    @check @runtime_ccall((:cusparseCreate, libcusparse()), cusparseStatus_t,
                 (Ptr{cusparseHandle_t},),
                 handle)
end

function cusparseDestroy(handle)
    @check @runtime_ccall((:cusparseDestroy, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t,),
                 handle)
end

function cusparseGetVersion(handle, version)
    @check @runtime_ccall((:cusparseGetVersion, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Ptr{Cint}),
                 handle, version)
end

function cusparseGetProperty(type, value)
    @check @runtime_ccall((:cusparseGetProperty, libcusparse()), cusparseStatus_t,
                 (libraryPropertyType, Ptr{Cint}),
                 type, value)
end

function cusparseGetErrorName(status)
    @runtime_ccall((:cusparseGetErrorName, libcusparse()), Cstring,
          (cusparseStatus_t,),
          status)
end

function cusparseGetErrorString(status)
    @runtime_ccall((:cusparseGetErrorString, libcusparse()), Cstring,
          (cusparseStatus_t,),
          status)
end

function cusparseSetStream(handle, streamId)
    @check @runtime_ccall((:cusparseSetStream, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, CUstream),
                 handle, streamId)
end

function cusparseGetStream(handle, streamId)
    @check @runtime_ccall((:cusparseGetStream, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Ptr{CUstream}),
                 handle, streamId)
end

function cusparseGetPointerMode(handle, mode)
    @check @runtime_ccall((:cusparseGetPointerMode, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Ptr{cusparsePointerMode_t}),
                 handle, mode)
end

function cusparseSetPointerMode(handle, mode)
    @check @runtime_ccall((:cusparseSetPointerMode, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparsePointerMode_t),
                 handle, mode)
end

function cusparseCreateMatDescr(descrA)
    @check @runtime_ccall((:cusparseCreateMatDescr, libcusparse()), cusparseStatus_t,
                 (Ptr{cusparseMatDescr_t},),
                 descrA)
end

function cusparseDestroyMatDescr(descrA)
    @check @runtime_ccall((:cusparseDestroyMatDescr, libcusparse()), cusparseStatus_t,
                 (cusparseMatDescr_t,),
                 descrA)
end

function cusparseCopyMatDescr(dest, src)
    @check @runtime_ccall((:cusparseCopyMatDescr, libcusparse()), cusparseStatus_t,
                 (cusparseMatDescr_t, cusparseMatDescr_t),
                 dest, src)
end

function cusparseSetMatType(descrA, type)
    @check @runtime_ccall((:cusparseSetMatType, libcusparse()), cusparseStatus_t,
                 (cusparseMatDescr_t, cusparseMatrixType_t),
                 descrA, type)
end

function cusparseGetMatType(descrA)
    @runtime_ccall((:cusparseGetMatType, libcusparse()), cusparseMatrixType_t,
          (cusparseMatDescr_t,),
          descrA)
end

function cusparseSetMatFillMode(descrA, fillMode)
    @check @runtime_ccall((:cusparseSetMatFillMode, libcusparse()), cusparseStatus_t,
                 (cusparseMatDescr_t, cusparseFillMode_t),
                 descrA, fillMode)
end

function cusparseGetMatFillMode(descrA)
    @runtime_ccall((:cusparseGetMatFillMode, libcusparse()), cusparseFillMode_t,
          (cusparseMatDescr_t,),
          descrA)
end

function cusparseSetMatDiagType(descrA, diagType)
    @check @runtime_ccall((:cusparseSetMatDiagType, libcusparse()), cusparseStatus_t,
                 (cusparseMatDescr_t, cusparseDiagType_t),
                 descrA, diagType)
end

function cusparseGetMatDiagType(descrA)
    @runtime_ccall((:cusparseGetMatDiagType, libcusparse()), cusparseDiagType_t,
          (cusparseMatDescr_t,),
          descrA)
end

function cusparseSetMatIndexBase(descrA, base)
    @check @runtime_ccall((:cusparseSetMatIndexBase, libcusparse()), cusparseStatus_t,
                 (cusparseMatDescr_t, cusparseIndexBase_t),
                 descrA, base)
end

function cusparseGetMatIndexBase(descrA)
    @runtime_ccall((:cusparseGetMatIndexBase, libcusparse()), cusparseIndexBase_t,
          (cusparseMatDescr_t,),
          descrA)
end

function cusparseCreateSolveAnalysisInfo(info)
    @check @runtime_ccall((:cusparseCreateSolveAnalysisInfo, libcusparse()), cusparseStatus_t,
                 (Ptr{cusparseSolveAnalysisInfo_t},),
                 info)
end

function cusparseDestroySolveAnalysisInfo(info)
    @check @runtime_ccall((:cusparseDestroySolveAnalysisInfo, libcusparse()), cusparseStatus_t,
                 (cusparseSolveAnalysisInfo_t,),
                 info)
end

function cusparseGetLevelInfo(handle, info, nlevels, levelPtr, levelInd)
    @check @runtime_ccall((:cusparseGetLevelInfo, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseSolveAnalysisInfo_t, Ptr{Cint},
                  Ptr{Ptr{Cint}}, Ptr{Ptr{Cint}}),
                 handle, info, nlevels, levelPtr, levelInd)
end

function cusparseCreateCsrsv2Info(info)
    @check @runtime_ccall((:cusparseCreateCsrsv2Info, libcusparse()), cusparseStatus_t,
                 (Ptr{csrsv2Info_t},),
                 info)
end

function cusparseDestroyCsrsv2Info(info)
    @check @runtime_ccall((:cusparseDestroyCsrsv2Info, libcusparse()), cusparseStatus_t,
                 (csrsv2Info_t,),
                 info)
end

function cusparseCreateCsric02Info(info)
    @check @runtime_ccall((:cusparseCreateCsric02Info, libcusparse()), cusparseStatus_t,
                 (Ptr{csric02Info_t},),
                 info)
end

function cusparseDestroyCsric02Info(info)
    @check @runtime_ccall((:cusparseDestroyCsric02Info, libcusparse()), cusparseStatus_t,
                 (csric02Info_t,),
                 info)
end

function cusparseCreateBsric02Info(info)
    @check @runtime_ccall((:cusparseCreateBsric02Info, libcusparse()), cusparseStatus_t,
                 (Ptr{bsric02Info_t},),
                 info)
end

function cusparseDestroyBsric02Info(info)
    @check @runtime_ccall((:cusparseDestroyBsric02Info, libcusparse()), cusparseStatus_t,
                 (bsric02Info_t,),
                 info)
end

function cusparseCreateCsrilu02Info(info)
    @check @runtime_ccall((:cusparseCreateCsrilu02Info, libcusparse()), cusparseStatus_t,
                 (Ptr{csrilu02Info_t},),
                 info)
end

function cusparseDestroyCsrilu02Info(info)
    @check @runtime_ccall((:cusparseDestroyCsrilu02Info, libcusparse()), cusparseStatus_t,
                 (csrilu02Info_t,),
                 info)
end

function cusparseCreateBsrilu02Info(info)
    @check @runtime_ccall((:cusparseCreateBsrilu02Info, libcusparse()), cusparseStatus_t,
                 (Ptr{bsrilu02Info_t},),
                 info)
end

function cusparseDestroyBsrilu02Info(info)
    @check @runtime_ccall((:cusparseDestroyBsrilu02Info, libcusparse()), cusparseStatus_t,
                 (bsrilu02Info_t,),
                 info)
end

function cusparseCreateBsrsv2Info(info)
    @check @runtime_ccall((:cusparseCreateBsrsv2Info, libcusparse()), cusparseStatus_t,
                 (Ptr{bsrsv2Info_t},),
                 info)
end

function cusparseDestroyBsrsv2Info(info)
    @check @runtime_ccall((:cusparseDestroyBsrsv2Info, libcusparse()), cusparseStatus_t,
                 (bsrsv2Info_t,),
                 info)
end

function cusparseCreateBsrsm2Info(info)
    @check @runtime_ccall((:cusparseCreateBsrsm2Info, libcusparse()), cusparseStatus_t,
                 (Ptr{bsrsm2Info_t},),
                 info)
end

function cusparseDestroyBsrsm2Info(info)
    @check @runtime_ccall((:cusparseDestroyBsrsm2Info, libcusparse()), cusparseStatus_t,
                 (bsrsm2Info_t,),
                 info)
end

function cusparseCreateHybMat(hybA)
    @check @runtime_ccall((:cusparseCreateHybMat, libcusparse()), cusparseStatus_t,
                 (Ptr{cusparseHybMat_t},),
                 hybA)
end

function cusparseDestroyHybMat(hybA)
    @check @runtime_ccall((:cusparseDestroyHybMat, libcusparse()), cusparseStatus_t,
                 (cusparseHybMat_t,),
                 hybA)
end

function cusparseCreateCsru2csrInfo(info)
    @check @runtime_ccall((:cusparseCreateCsru2csrInfo, libcusparse()), cusparseStatus_t,
                 (Ptr{csru2csrInfo_t},),
                 info)
end

function cusparseDestroyCsru2csrInfo(info)
    @check @runtime_ccall((:cusparseDestroyCsru2csrInfo, libcusparse()), cusparseStatus_t,
                 (csru2csrInfo_t,),
                 info)
end

function cusparseCreateColorInfo(info)
    @check @runtime_ccall((:cusparseCreateColorInfo, libcusparse()), cusparseStatus_t,
                 (Ptr{cusparseColorInfo_t},),
                 info)
end

function cusparseDestroyColorInfo(info)
    @check @runtime_ccall((:cusparseDestroyColorInfo, libcusparse()), cusparseStatus_t,
                 (cusparseColorInfo_t,),
                 info)
end

function cusparseSetColorAlgs(info, alg)
    @check @runtime_ccall((:cusparseSetColorAlgs, libcusparse()), cusparseStatus_t,
                 (cusparseColorInfo_t, cusparseColorAlg_t),
                 info, alg)
end

function cusparseGetColorAlgs(info, alg)
    @check @runtime_ccall((:cusparseGetColorAlgs, libcusparse()), cusparseStatus_t,
                 (cusparseColorInfo_t, Ptr{cusparseColorAlg_t}),
                 info, alg)
end

function cusparseCreatePruneInfo(info)
    @check @runtime_ccall((:cusparseCreatePruneInfo, libcusparse()), cusparseStatus_t,
                 (Ptr{pruneInfo_t},),
                 info)
end

function cusparseDestroyPruneInfo(info)
    @check @runtime_ccall((:cusparseDestroyPruneInfo, libcusparse()), cusparseStatus_t,
                 (pruneInfo_t,),
                 info)
end

function cusparseSaxpyi(handle, nnz, alpha, xVal, xInd, y, idxBase)
    @check @runtime_ccall((:cusparseSaxpyi, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Ptr{Cfloat}, CuPtr{Cfloat}, CuPtr{Cint},
                  CuPtr{Cfloat}, cusparseIndexBase_t),
                 handle, nnz, alpha, xVal, xInd, y, idxBase)
end

function cusparseDaxpyi(handle, nnz, alpha, xVal, xInd, y, idxBase)
    @check @runtime_ccall((:cusparseDaxpyi, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Ptr{Cdouble}, CuPtr{Cdouble}, CuPtr{Cint},
                  CuPtr{Cdouble}, cusparseIndexBase_t),
                 handle, nnz, alpha, xVal, xInd, y, idxBase)
end

function cusparseCaxpyi(handle, nnz, alpha, xVal, xInd, y, idxBase)
    @check @runtime_ccall((:cusparseCaxpyi, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Ptr{cuComplex}, CuPtr{cuComplex}, CuPtr{Cint},
                  CuPtr{cuComplex}, cusparseIndexBase_t),
                 handle, nnz, alpha, xVal, xInd, y, idxBase)
end

function cusparseZaxpyi(handle, nnz, alpha, xVal, xInd, y, idxBase)
    @check @runtime_ccall((:cusparseZaxpyi, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Ptr{cuDoubleComplex}, CuPtr{cuDoubleComplex},
                  CuPtr{Cint}, CuPtr{cuDoubleComplex}, cusparseIndexBase_t),
                 handle, nnz, alpha, xVal, xInd, y, idxBase)
end

function cusparseSdoti(handle, nnz, xVal, xInd, y, resultDevHostPtr, idxBase)
    @check @runtime_ccall((:cusparseSdoti, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, CuPtr{Cfloat}, CuPtr{Cint}, CuPtr{Cfloat},
                  PtrOrCuPtr{Cfloat}, cusparseIndexBase_t),
                 handle, nnz, xVal, xInd, y, resultDevHostPtr, idxBase)
end

function cusparseDdoti(handle, nnz, xVal, xInd, y, resultDevHostPtr, idxBase)
    @check @runtime_ccall((:cusparseDdoti, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, CuPtr{Cdouble}, CuPtr{Cint}, CuPtr{Cdouble},
                  PtrOrCuPtr{Cdouble}, cusparseIndexBase_t),
                 handle, nnz, xVal, xInd, y, resultDevHostPtr, idxBase)
end

function cusparseCdoti(handle, nnz, xVal, xInd, y, resultDevHostPtr, idxBase)
    @check @runtime_ccall((:cusparseCdoti, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, CuPtr{cuComplex}, CuPtr{Cint}, CuPtr{cuComplex},
                  PtrOrCuPtr{cuComplex}, cusparseIndexBase_t),
                 handle, nnz, xVal, xInd, y, resultDevHostPtr, idxBase)
end

function cusparseZdoti(handle, nnz, xVal, xInd, y, resultDevHostPtr, idxBase)
    @check @runtime_ccall((:cusparseZdoti, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, CuPtr{cuDoubleComplex}, CuPtr{Cint},
                  CuPtr{cuDoubleComplex}, PtrOrCuPtr{cuDoubleComplex}, cusparseIndexBase_t),
                 handle, nnz, xVal, xInd, y, resultDevHostPtr, idxBase)
end

function cusparseCdotci(handle, nnz, xVal, xInd, y, resultDevHostPtr, idxBase)
    @check @runtime_ccall((:cusparseCdotci, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, CuPtr{cuComplex}, CuPtr{Cint}, CuPtr{cuComplex},
                  PtrOrCuPtr{cuComplex}, cusparseIndexBase_t),
                 handle, nnz, xVal, xInd, y, resultDevHostPtr, idxBase)
end

function cusparseZdotci(handle, nnz, xVal, xInd, y, resultDevHostPtr, idxBase)
    @check @runtime_ccall((:cusparseZdotci, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, CuPtr{cuDoubleComplex}, CuPtr{Cint},
                  CuPtr{cuDoubleComplex}, PtrOrCuPtr{cuDoubleComplex}, cusparseIndexBase_t),
                 handle, nnz, xVal, xInd, y, resultDevHostPtr, idxBase)
end

function cusparseSgthr(handle, nnz, y, xVal, xInd, idxBase)
    @check @runtime_ccall((:cusparseSgthr, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, CuPtr{Cfloat}, CuPtr{Cfloat}, CuPtr{Cint},
                  cusparseIndexBase_t),
                 handle, nnz, y, xVal, xInd, idxBase)
end

function cusparseDgthr(handle, nnz, y, xVal, xInd, idxBase)
    @check @runtime_ccall((:cusparseDgthr, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, CuPtr{Cdouble}, CuPtr{Cdouble}, CuPtr{Cint},
                  cusparseIndexBase_t),
                 handle, nnz, y, xVal, xInd, idxBase)
end

function cusparseCgthr(handle, nnz, y, xVal, xInd, idxBase)
    @check @runtime_ccall((:cusparseCgthr, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, CuPtr{cuComplex}, CuPtr{cuComplex}, CuPtr{Cint},
                  cusparseIndexBase_t),
                 handle, nnz, y, xVal, xInd, idxBase)
end

function cusparseZgthr(handle, nnz, y, xVal, xInd, idxBase)
    @check @runtime_ccall((:cusparseZgthr, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, CuPtr{cuDoubleComplex}, CuPtr{cuDoubleComplex},
                  CuPtr{Cint}, cusparseIndexBase_t),
                 handle, nnz, y, xVal, xInd, idxBase)
end

function cusparseSgthrz(handle, nnz, y, xVal, xInd, idxBase)
    @check @runtime_ccall((:cusparseSgthrz, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, CuPtr{Cfloat}, CuPtr{Cfloat}, CuPtr{Cint},
                  cusparseIndexBase_t),
                 handle, nnz, y, xVal, xInd, idxBase)
end

function cusparseDgthrz(handle, nnz, y, xVal, xInd, idxBase)
    @check @runtime_ccall((:cusparseDgthrz, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, CuPtr{Cdouble}, CuPtr{Cdouble}, CuPtr{Cint},
                  cusparseIndexBase_t),
                 handle, nnz, y, xVal, xInd, idxBase)
end

function cusparseCgthrz(handle, nnz, y, xVal, xInd, idxBase)
    @check @runtime_ccall((:cusparseCgthrz, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, CuPtr{cuComplex}, CuPtr{cuComplex}, CuPtr{Cint},
                  cusparseIndexBase_t),
                 handle, nnz, y, xVal, xInd, idxBase)
end

function cusparseZgthrz(handle, nnz, y, xVal, xInd, idxBase)
    @check @runtime_ccall((:cusparseZgthrz, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, CuPtr{cuDoubleComplex}, CuPtr{cuDoubleComplex},
                  CuPtr{Cint}, cusparseIndexBase_t),
                 handle, nnz, y, xVal, xInd, idxBase)
end

function cusparseSsctr(handle, nnz, xVal, xInd, y, idxBase)
    @check @runtime_ccall((:cusparseSsctr, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, CuPtr{Cfloat}, CuPtr{Cint}, CuPtr{Cfloat},
                  cusparseIndexBase_t),
                 handle, nnz, xVal, xInd, y, idxBase)
end

function cusparseDsctr(handle, nnz, xVal, xInd, y, idxBase)
    @check @runtime_ccall((:cusparseDsctr, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, CuPtr{Cdouble}, CuPtr{Cint}, CuPtr{Cdouble},
                  cusparseIndexBase_t),
                 handle, nnz, xVal, xInd, y, idxBase)
end

function cusparseCsctr(handle, nnz, xVal, xInd, y, idxBase)
    @check @runtime_ccall((:cusparseCsctr, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, CuPtr{cuComplex}, CuPtr{Cint}, CuPtr{cuComplex},
                  cusparseIndexBase_t),
                 handle, nnz, xVal, xInd, y, idxBase)
end

function cusparseZsctr(handle, nnz, xVal, xInd, y, idxBase)
    @check @runtime_ccall((:cusparseZsctr, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, CuPtr{cuDoubleComplex}, CuPtr{Cint},
                  CuPtr{cuDoubleComplex}, cusparseIndexBase_t),
                 handle, nnz, xVal, xInd, y, idxBase)
end

function cusparseSroti(handle, nnz, xVal, xInd, y, c, s, idxBase)
    @check @runtime_ccall((:cusparseSroti, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, CuPtr{Cfloat}, CuPtr{Cint}, CuPtr{Cfloat},
                  Ptr{Cfloat}, Ptr{Cfloat}, cusparseIndexBase_t),
                 handle, nnz, xVal, xInd, y, c, s, idxBase)
end

function cusparseDroti(handle, nnz, xVal, xInd, y, c, s, idxBase)
    @check @runtime_ccall((:cusparseDroti, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, CuPtr{Cdouble}, CuPtr{Cint}, CuPtr{Cdouble},
                  Ptr{Cdouble}, Ptr{Cdouble}, cusparseIndexBase_t),
                 handle, nnz, xVal, xInd, y, c, s, idxBase)
end

function cusparseSgemvi(handle, transA, m, n, alpha, A, lda, nnz, xVal, xInd, beta, y,
                        idxBase, pBuffer)
    @check @runtime_ccall((:cusparseSgemvi, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseOperation_t, Cint, Cint, Ptr{Cfloat},
                  CuPtr{Cfloat}, Cint, Cint, CuPtr{Cfloat}, CuPtr{Cint}, Ptr{Cfloat},
                  CuPtr{Cfloat}, cusparseIndexBase_t, CuPtr{Cvoid}),
                 handle, transA, m, n, alpha, A, lda, nnz, xVal, xInd, beta, y, idxBase,
                 pBuffer)
end

function cusparseSgemvi_bufferSize(handle, transA, m, n, nnz, pBufferSize)
    @check @runtime_ccall((:cusparseSgemvi_bufferSize, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseOperation_t, Cint, Cint, Cint, Ptr{Cint}),
                 handle, transA, m, n, nnz, pBufferSize)
end

function cusparseDgemvi(handle, transA, m, n, alpha, A, lda, nnz, xVal, xInd, beta, y,
                        idxBase, pBuffer)
    @check @runtime_ccall((:cusparseDgemvi, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseOperation_t, Cint, Cint, Ptr{Cdouble},
                  CuPtr{Cdouble}, Cint, Cint, CuPtr{Cdouble}, CuPtr{Cint}, Ptr{Cdouble},
                  CuPtr{Cdouble}, cusparseIndexBase_t, CuPtr{Cvoid}),
                 handle, transA, m, n, alpha, A, lda, nnz, xVal, xInd, beta, y, idxBase,
                 pBuffer)
end

function cusparseDgemvi_bufferSize(handle, transA, m, n, nnz, pBufferSize)
    @check @runtime_ccall((:cusparseDgemvi_bufferSize, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseOperation_t, Cint, Cint, Cint, Ptr{Cint}),
                 handle, transA, m, n, nnz, pBufferSize)
end

function cusparseCgemvi(handle, transA, m, n, alpha, A, lda, nnz, xVal, xInd, beta, y,
                        idxBase, pBuffer)
    @check @runtime_ccall((:cusparseCgemvi, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseOperation_t, Cint, Cint, Ptr{cuComplex},
                  CuPtr{cuComplex}, Cint, Cint, CuPtr{cuComplex}, CuPtr{Cint},
                  Ptr{cuComplex}, CuPtr{cuComplex}, cusparseIndexBase_t, CuPtr{Cvoid}),
                 handle, transA, m, n, alpha, A, lda, nnz, xVal, xInd, beta, y, idxBase,
                 pBuffer)
end

function cusparseCgemvi_bufferSize(handle, transA, m, n, nnz, pBufferSize)
    @check @runtime_ccall((:cusparseCgemvi_bufferSize, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseOperation_t, Cint, Cint, Cint, Ptr{Cint}),
                 handle, transA, m, n, nnz, pBufferSize)
end

function cusparseZgemvi(handle, transA, m, n, alpha, A, lda, nnz, xVal, xInd, beta, y,
                        idxBase, pBuffer)
    @check @runtime_ccall((:cusparseZgemvi, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseOperation_t, Cint, Cint, Ptr{cuDoubleComplex},
                  CuPtr{cuDoubleComplex}, Cint, Cint, CuPtr{cuDoubleComplex}, CuPtr{Cint},
                  Ptr{cuDoubleComplex}, CuPtr{cuDoubleComplex}, cusparseIndexBase_t,
                  CuPtr{Cvoid}),
                 handle, transA, m, n, alpha, A, lda, nnz, xVal, xInd, beta, y, idxBase,
                 pBuffer)
end

function cusparseZgemvi_bufferSize(handle, transA, m, n, nnz, pBufferSize)
    @check @runtime_ccall((:cusparseZgemvi_bufferSize, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseOperation_t, Cint, Cint, Cint, Ptr{Cint}),
                 handle, transA, m, n, nnz, pBufferSize)
end

function cusparseScsrmv(handle, transA, m, n, nnz, alpha, descrA, csrSortedValA,
                        csrSortedRowPtrA, csrSortedColIndA, x, beta, y)
    @check @runtime_ccall((:cusparseScsrmv, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseOperation_t, Cint, Cint, Cint, Ptr{Cfloat},
                  cusparseMatDescr_t, CuPtr{Cfloat}, CuPtr{Cint}, CuPtr{Cint},
                  CuPtr{Cfloat}, Ptr{Cfloat}, CuPtr{Cfloat}),
                 handle, transA, m, n, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA,
                 csrSortedColIndA, x, beta, y)
end

function cusparseDcsrmv(handle, transA, m, n, nnz, alpha, descrA, csrSortedValA,
                        csrSortedRowPtrA, csrSortedColIndA, x, beta, y)
    @check @runtime_ccall((:cusparseDcsrmv, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseOperation_t, Cint, Cint, Cint, Ptr{Cdouble},
                  cusparseMatDescr_t, CuPtr{Cdouble}, CuPtr{Cint}, CuPtr{Cint},
                  CuPtr{Cdouble}, Ptr{Cdouble}, CuPtr{Cdouble}),
                 handle, transA, m, n, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA,
                 csrSortedColIndA, x, beta, y)
end

function cusparseCcsrmv(handle, transA, m, n, nnz, alpha, descrA, csrSortedValA,
                        csrSortedRowPtrA, csrSortedColIndA, x, beta, y)
    @check @runtime_ccall((:cusparseCcsrmv, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseOperation_t, Cint, Cint, Cint, Ptr{cuComplex},
                  cusparseMatDescr_t, CuPtr{cuComplex}, CuPtr{Cint}, CuPtr{Cint},
                  CuPtr{cuComplex}, Ptr{cuComplex}, CuPtr{cuComplex}),
                 handle, transA, m, n, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA,
                 csrSortedColIndA, x, beta, y)
end

function cusparseZcsrmv(handle, transA, m, n, nnz, alpha, descrA, csrSortedValA,
                        csrSortedRowPtrA, csrSortedColIndA, x, beta, y)
    @check @runtime_ccall((:cusparseZcsrmv, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseOperation_t, Cint, Cint, Cint,
                  Ptr{cuDoubleComplex}, cusparseMatDescr_t, CuPtr{cuDoubleComplex},
                  CuPtr{Cint}, CuPtr{Cint}, CuPtr{cuDoubleComplex}, Ptr{cuDoubleComplex},
                  CuPtr{cuDoubleComplex}),
                 handle, transA, m, n, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA,
                 csrSortedColIndA, x, beta, y)
end

function cusparseCsrmvEx_bufferSize(handle, alg, transA, m, n, nnz, alpha, alphatype,
                                    descrA, csrValA, csrValAtype, csrRowPtrA, csrColIndA,
                                    x, xtype, beta, betatype, y, ytype, executiontype,
                                    bufferSizeInBytes)
    @check @runtime_ccall((:cusparseCsrmvEx_bufferSize, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseAlgMode_t, cusparseOperation_t, Cint, Cint,
                  Cint, Ptr{Cvoid}, cudaDataType, cusparseMatDescr_t, CuPtr{Cvoid},
                  cudaDataType, CuPtr{Cint}, CuPtr{Cint}, CuPtr{Cvoid}, cudaDataType,
                  Ptr{Cvoid}, cudaDataType, CuPtr{Cvoid}, cudaDataType, cudaDataType,
                  Ptr{Csize_t}),
                 handle, alg, transA, m, n, nnz, alpha, alphatype, descrA, csrValA,
                 csrValAtype, csrRowPtrA, csrColIndA, x, xtype, beta, betatype, y, ytype,
                 executiontype, bufferSizeInBytes)
end

function cusparseCsrmvEx(handle, alg, transA, m, n, nnz, alpha, alphatype, descrA, csrValA,
                         csrValAtype, csrRowPtrA, csrColIndA, x, xtype, beta, betatype, y,
                         ytype, executiontype, buffer)
    @check @runtime_ccall((:cusparseCsrmvEx, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseAlgMode_t, cusparseOperation_t, Cint, Cint,
                  Cint, Ptr{Cvoid}, cudaDataType, cusparseMatDescr_t, CuPtr{Cvoid},
                  cudaDataType, CuPtr{Cint}, CuPtr{Cint}, CuPtr{Cvoid}, cudaDataType,
                  Ptr{Cvoid}, cudaDataType, CuPtr{Cvoid}, cudaDataType, cudaDataType,
                  CuPtr{Cvoid}),
                 handle, alg, transA, m, n, nnz, alpha, alphatype, descrA, csrValA,
                 csrValAtype, csrRowPtrA, csrColIndA, x, xtype, beta, betatype, y, ytype,
                 executiontype, buffer)
end

function cusparseScsrmv_mp(handle, transA, m, n, nnz, alpha, descrA, csrSortedValA,
                           csrSortedRowPtrA, csrSortedColIndA, x, beta, y)
    @check @runtime_ccall((:cusparseScsrmv_mp, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseOperation_t, Cint, Cint, Cint, Ptr{Cfloat},
                  cusparseMatDescr_t, CuPtr{Cfloat}, CuPtr{Cint}, CuPtr{Cint},
                  CuPtr{Cfloat}, Ptr{Cfloat}, CuPtr{Cfloat}),
                 handle, transA, m, n, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA,
                 csrSortedColIndA, x, beta, y)
end

function cusparseDcsrmv_mp(handle, transA, m, n, nnz, alpha, descrA, csrSortedValA,
                           csrSortedRowPtrA, csrSortedColIndA, x, beta, y)
    @check @runtime_ccall((:cusparseDcsrmv_mp, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseOperation_t, Cint, Cint, Cint, Ptr{Cdouble},
                  cusparseMatDescr_t, CuPtr{Cdouble}, CuPtr{Cint}, CuPtr{Cint},
                  CuPtr{Cdouble}, Ptr{Cdouble}, CuPtr{Cdouble}),
                 handle, transA, m, n, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA,
                 csrSortedColIndA, x, beta, y)
end

function cusparseCcsrmv_mp(handle, transA, m, n, nnz, alpha, descrA, csrSortedValA,
                           csrSortedRowPtrA, csrSortedColIndA, x, beta, y)
    @check @runtime_ccall((:cusparseCcsrmv_mp, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseOperation_t, Cint, Cint, Cint, Ptr{cuComplex},
                  cusparseMatDescr_t, CuPtr{cuComplex}, CuPtr{Cint}, CuPtr{Cint},
                  CuPtr{cuComplex}, Ptr{cuComplex}, CuPtr{cuComplex}),
                 handle, transA, m, n, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA,
                 csrSortedColIndA, x, beta, y)
end

function cusparseZcsrmv_mp(handle, transA, m, n, nnz, alpha, descrA, csrSortedValA,
                           csrSortedRowPtrA, csrSortedColIndA, x, beta, y)
    @check @runtime_ccall((:cusparseZcsrmv_mp, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseOperation_t, Cint, Cint, Cint,
                  Ptr{cuDoubleComplex}, cusparseMatDescr_t, CuPtr{cuDoubleComplex},
                  CuPtr{Cint}, CuPtr{Cint}, CuPtr{cuDoubleComplex}, Ptr{cuDoubleComplex},
                  CuPtr{cuDoubleComplex}),
                 handle, transA, m, n, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA,
                 csrSortedColIndA, x, beta, y)
end

function cusparseShybmv(handle, transA, alpha, descrA, hybA, x, beta, y)
    @check @runtime_ccall((:cusparseShybmv, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseOperation_t, Ptr{Cfloat}, cusparseMatDescr_t,
                  cusparseHybMat_t, CuPtr{Cfloat}, Ptr{Cfloat}, CuPtr{Cfloat}),
                 handle, transA, alpha, descrA, hybA, x, beta, y)
end

function cusparseDhybmv(handle, transA, alpha, descrA, hybA, x, beta, y)
    @check @runtime_ccall((:cusparseDhybmv, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseOperation_t, Ptr{Cdouble}, cusparseMatDescr_t,
                  cusparseHybMat_t, CuPtr{Cdouble}, Ptr{Cdouble}, CuPtr{Cdouble}),
                 handle, transA, alpha, descrA, hybA, x, beta, y)
end

function cusparseChybmv(handle, transA, alpha, descrA, hybA, x, beta, y)
    @check @runtime_ccall((:cusparseChybmv, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseOperation_t, Ptr{cuComplex},
                  cusparseMatDescr_t, cusparseHybMat_t, CuPtr{cuComplex}, Ptr{cuComplex},
                  CuPtr{cuComplex}),
                 handle, transA, alpha, descrA, hybA, x, beta, y)
end

function cusparseZhybmv(handle, transA, alpha, descrA, hybA, x, beta, y)
    @check @runtime_ccall((:cusparseZhybmv, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseOperation_t, Ptr{cuDoubleComplex},
                  cusparseMatDescr_t, cusparseHybMat_t, CuPtr{cuDoubleComplex},
                  Ptr{cuDoubleComplex}, CuPtr{cuDoubleComplex}),
                 handle, transA, alpha, descrA, hybA, x, beta, y)
end

function cusparseSbsrmv(handle, dirA, transA, mb, nb, nnzb, alpha, descrA, bsrSortedValA,
                        bsrSortedRowPtrA, bsrSortedColIndA, blockDim, x, beta, y)
    @check @runtime_ccall((:cusparseSbsrmv, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, Cint, Cint,
                  Cint, Ptr{Cfloat}, cusparseMatDescr_t, CuPtr{Cfloat}, CuPtr{Cint},
                  CuPtr{Cint}, Cint, CuPtr{Cfloat}, Ptr{Cfloat}, CuPtr{Cfloat}),
                 handle, dirA, transA, mb, nb, nnzb, alpha, descrA, bsrSortedValA,
                 bsrSortedRowPtrA, bsrSortedColIndA, blockDim, x, beta, y)
end

function cusparseDbsrmv(handle, dirA, transA, mb, nb, nnzb, alpha, descrA, bsrSortedValA,
                        bsrSortedRowPtrA, bsrSortedColIndA, blockDim, x, beta, y)
    @check @runtime_ccall((:cusparseDbsrmv, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, Cint, Cint,
                  Cint, Ptr{Cdouble}, cusparseMatDescr_t, CuPtr{Cdouble}, CuPtr{Cint},
                  CuPtr{Cint}, Cint, CuPtr{Cdouble}, Ptr{Cdouble}, CuPtr{Cdouble}),
                 handle, dirA, transA, mb, nb, nnzb, alpha, descrA, bsrSortedValA,
                 bsrSortedRowPtrA, bsrSortedColIndA, blockDim, x, beta, y)
end

function cusparseCbsrmv(handle, dirA, transA, mb, nb, nnzb, alpha, descrA, bsrSortedValA,
                        bsrSortedRowPtrA, bsrSortedColIndA, blockDim, x, beta, y)
    @check @runtime_ccall((:cusparseCbsrmv, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, Cint, Cint,
                  Cint, Ptr{cuComplex}, cusparseMatDescr_t, CuPtr{cuComplex}, CuPtr{Cint},
                  CuPtr{Cint}, Cint, CuPtr{cuComplex}, Ptr{cuComplex}, CuPtr{cuComplex}),
                 handle, dirA, transA, mb, nb, nnzb, alpha, descrA, bsrSortedValA,
                 bsrSortedRowPtrA, bsrSortedColIndA, blockDim, x, beta, y)
end

function cusparseZbsrmv(handle, dirA, transA, mb, nb, nnzb, alpha, descrA, bsrSortedValA,
                        bsrSortedRowPtrA, bsrSortedColIndA, blockDim, x, beta, y)
    @check @runtime_ccall((:cusparseZbsrmv, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, Cint, Cint,
                  Cint, Ptr{cuDoubleComplex}, cusparseMatDescr_t, CuPtr{cuDoubleComplex},
                  CuPtr{Cint}, CuPtr{Cint}, Cint, CuPtr{cuDoubleComplex},
                  Ptr{cuDoubleComplex}, CuPtr{cuDoubleComplex}),
                 handle, dirA, transA, mb, nb, nnzb, alpha, descrA, bsrSortedValA,
                 bsrSortedRowPtrA, bsrSortedColIndA, blockDim, x, beta, y)
end

function cusparseSbsrxmv(handle, dirA, transA, sizeOfMask, mb, nb, nnzb, alpha, descrA,
                         bsrSortedValA, bsrSortedMaskPtrA, bsrSortedRowPtrA,
                         bsrSortedEndPtrA, bsrSortedColIndA, blockDim, x, beta, y)
    @check @runtime_ccall((:cusparseSbsrxmv, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, Cint, Cint,
                  Cint, Cint, Ptr{Cfloat}, cusparseMatDescr_t, CuPtr{Cfloat}, CuPtr{Cint},
                  CuPtr{Cint}, CuPtr{Cint}, CuPtr{Cint}, Cint, CuPtr{Cfloat}, Ptr{Cfloat},
                  CuPtr{Cfloat}),
                 handle, dirA, transA, sizeOfMask, mb, nb, nnzb, alpha, descrA,
                 bsrSortedValA, bsrSortedMaskPtrA, bsrSortedRowPtrA, bsrSortedEndPtrA,
                 bsrSortedColIndA, blockDim, x, beta, y)
end

function cusparseDbsrxmv(handle, dirA, transA, sizeOfMask, mb, nb, nnzb, alpha, descrA,
                         bsrSortedValA, bsrSortedMaskPtrA, bsrSortedRowPtrA,
                         bsrSortedEndPtrA, bsrSortedColIndA, blockDim, x, beta, y)
    @check @runtime_ccall((:cusparseDbsrxmv, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, Cint, Cint,
                  Cint, Cint, Ptr{Cdouble}, cusparseMatDescr_t, CuPtr{Cdouble},
                  CuPtr{Cint}, CuPtr{Cint}, CuPtr{Cint}, CuPtr{Cint}, Cint, CuPtr{Cdouble},
                  Ptr{Cdouble}, CuPtr{Cdouble}),
                 handle, dirA, transA, sizeOfMask, mb, nb, nnzb, alpha, descrA,
                 bsrSortedValA, bsrSortedMaskPtrA, bsrSortedRowPtrA, bsrSortedEndPtrA,
                 bsrSortedColIndA, blockDim, x, beta, y)
end

function cusparseCbsrxmv(handle, dirA, transA, sizeOfMask, mb, nb, nnzb, alpha, descrA,
                         bsrSortedValA, bsrSortedMaskPtrA, bsrSortedRowPtrA,
                         bsrSortedEndPtrA, bsrSortedColIndA, blockDim, x, beta, y)
    @check @runtime_ccall((:cusparseCbsrxmv, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, Cint, Cint,
                  Cint, Cint, Ptr{cuComplex}, cusparseMatDescr_t, CuPtr{cuComplex},
                  CuPtr{Cint}, CuPtr{Cint}, CuPtr{Cint}, CuPtr{Cint}, Cint,
                  CuPtr{cuComplex}, Ptr{cuComplex}, CuPtr{cuComplex}),
                 handle, dirA, transA, sizeOfMask, mb, nb, nnzb, alpha, descrA,
                 bsrSortedValA, bsrSortedMaskPtrA, bsrSortedRowPtrA, bsrSortedEndPtrA,
                 bsrSortedColIndA, blockDim, x, beta, y)
end

function cusparseZbsrxmv(handle, dirA, transA, sizeOfMask, mb, nb, nnzb, alpha, descrA,
                         bsrSortedValA, bsrSortedMaskPtrA, bsrSortedRowPtrA,
                         bsrSortedEndPtrA, bsrSortedColIndA, blockDim, x, beta, y)
    @check @runtime_ccall((:cusparseZbsrxmv, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, Cint, Cint,
                  Cint, Cint, Ptr{cuDoubleComplex}, cusparseMatDescr_t,
                  CuPtr{cuDoubleComplex}, CuPtr{Cint}, CuPtr{Cint}, CuPtr{Cint},
                  CuPtr{Cint}, Cint, CuPtr{cuDoubleComplex}, Ptr{cuDoubleComplex},
                  CuPtr{cuDoubleComplex}),
                 handle, dirA, transA, sizeOfMask, mb, nb, nnzb, alpha, descrA,
                 bsrSortedValA, bsrSortedMaskPtrA, bsrSortedRowPtrA, bsrSortedEndPtrA,
                 bsrSortedColIndA, blockDim, x, beta, y)
end

function cusparseCsrsv_analysisEx(handle, transA, m, nnz, descrA, csrSortedValA,
                                  csrSortedValAtype, csrSortedRowPtrA, csrSortedColIndA,
                                  info, executiontype)
    @check @runtime_ccall((:cusparseCsrsv_analysisEx, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseOperation_t, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{Cvoid}, cudaDataType, CuPtr{Cint}, CuPtr{Cint},
                  cusparseSolveAnalysisInfo_t, cudaDataType),
                 handle, transA, m, nnz, descrA, csrSortedValA, csrSortedValAtype,
                 csrSortedRowPtrA, csrSortedColIndA, info, executiontype)
end

function cusparseScsrsv_analysis(handle, transA, m, nnz, descrA, csrSortedValA,
                                 csrSortedRowPtrA, csrSortedColIndA, info)
    @check @runtime_ccall((:cusparseScsrsv_analysis, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseOperation_t, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{Cfloat}, CuPtr{Cint}, CuPtr{Cint}, cusparseSolveAnalysisInfo_t),
                 handle, transA, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA,
                 csrSortedColIndA, info)
end

function cusparseDcsrsv_analysis(handle, transA, m, nnz, descrA, csrSortedValA,
                                 csrSortedRowPtrA, csrSortedColIndA, info)
    @check @runtime_ccall((:cusparseDcsrsv_analysis, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseOperation_t, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{Cdouble}, CuPtr{Cint}, CuPtr{Cint}, cusparseSolveAnalysisInfo_t),
                 handle, transA, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA,
                 csrSortedColIndA, info)
end

function cusparseCcsrsv_analysis(handle, transA, m, nnz, descrA, csrSortedValA,
                                 csrSortedRowPtrA, csrSortedColIndA, info)
    @check @runtime_ccall((:cusparseCcsrsv_analysis, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseOperation_t, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{cuComplex}, CuPtr{Cint}, CuPtr{Cint}, cusparseSolveAnalysisInfo_t),
                 handle, transA, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA,
                 csrSortedColIndA, info)
end

function cusparseZcsrsv_analysis(handle, transA, m, nnz, descrA, csrSortedValA,
                                 csrSortedRowPtrA, csrSortedColIndA, info)
    @check @runtime_ccall((:cusparseZcsrsv_analysis, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseOperation_t, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{cuDoubleComplex}, CuPtr{Cint}, CuPtr{Cint},
                  cusparseSolveAnalysisInfo_t),
                 handle, transA, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA,
                 csrSortedColIndA, info)
end

function cusparseCsrsv_solveEx(handle, transA, m, alpha, alphatype, descrA, csrSortedValA,
                               csrSortedValAtype, csrSortedRowPtrA, csrSortedColIndA, info,
                               f, ftype, x, xtype, executiontype)
    @check @runtime_ccall((:cusparseCsrsv_solveEx, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseOperation_t, Cint, Ptr{Cvoid}, cudaDataType,
                  cusparseMatDescr_t, CuPtr{Cvoid}, cudaDataType, CuPtr{Cint}, CuPtr{Cint},
                  cusparseSolveAnalysisInfo_t, CuPtr{Cvoid}, cudaDataType, CuPtr{Cvoid},
                  cudaDataType, cudaDataType),
                 handle, transA, m, alpha, alphatype, descrA, csrSortedValA,
                 csrSortedValAtype, csrSortedRowPtrA, csrSortedColIndA, info, f, ftype, x,
                 xtype, executiontype)
end

function cusparseScsrsv_solve(handle, transA, m, alpha, descrA, csrSortedValA,
                              csrSortedRowPtrA, csrSortedColIndA, info, f, x)
    @check @runtime_ccall((:cusparseScsrsv_solve, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseOperation_t, Cint, Ptr{Cfloat},
                  cusparseMatDescr_t, CuPtr{Cfloat}, CuPtr{Cint}, CuPtr{Cint},
                  cusparseSolveAnalysisInfo_t, CuPtr{Cfloat}, CuPtr{Cfloat}),
                 handle, transA, m, alpha, descrA, csrSortedValA, csrSortedRowPtrA,
                 csrSortedColIndA, info, f, x)
end

function cusparseDcsrsv_solve(handle, transA, m, alpha, descrA, csrSortedValA,
                              csrSortedRowPtrA, csrSortedColIndA, info, f, x)
    @check @runtime_ccall((:cusparseDcsrsv_solve, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseOperation_t, Cint, Ptr{Cdouble},
                  cusparseMatDescr_t, CuPtr{Cdouble}, CuPtr{Cint}, CuPtr{Cint},
                  cusparseSolveAnalysisInfo_t, CuPtr{Cdouble}, CuPtr{Cdouble}),
                 handle, transA, m, alpha, descrA, csrSortedValA, csrSortedRowPtrA,
                 csrSortedColIndA, info, f, x)
end

function cusparseCcsrsv_solve(handle, transA, m, alpha, descrA, csrSortedValA,
                              csrSortedRowPtrA, csrSortedColIndA, info, f, x)
    @check @runtime_ccall((:cusparseCcsrsv_solve, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseOperation_t, Cint, Ptr{cuComplex},
                  cusparseMatDescr_t, CuPtr{cuComplex}, CuPtr{Cint}, CuPtr{Cint},
                  cusparseSolveAnalysisInfo_t, CuPtr{cuComplex}, CuPtr{cuComplex}),
                 handle, transA, m, alpha, descrA, csrSortedValA, csrSortedRowPtrA,
                 csrSortedColIndA, info, f, x)
end

function cusparseZcsrsv_solve(handle, transA, m, alpha, descrA, csrSortedValA,
                              csrSortedRowPtrA, csrSortedColIndA, info, f, x)
    @check @runtime_ccall((:cusparseZcsrsv_solve, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseOperation_t, Cint, Ptr{cuDoubleComplex},
                  cusparseMatDescr_t, CuPtr{cuDoubleComplex}, CuPtr{Cint}, CuPtr{Cint},
                  cusparseSolveAnalysisInfo_t, CuPtr{cuDoubleComplex},
                  CuPtr{cuDoubleComplex}),
                 handle, transA, m, alpha, descrA, csrSortedValA, csrSortedRowPtrA,
                 csrSortedColIndA, info, f, x)
end

function cusparseXcsrsv2_zeroPivot(handle, info, position)
    @check @runtime_ccall((:cusparseXcsrsv2_zeroPivot, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, csrsv2Info_t, Ptr{Cint}),
                 handle, info, position)
end

function cusparseScsrsv2_bufferSize(handle, transA, m, nnz, descrA, csrSortedValA,
                                    csrSortedRowPtrA, csrSortedColIndA, info,
                                    pBufferSizeInBytes)
    @check @runtime_ccall((:cusparseScsrsv2_bufferSize, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseOperation_t, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{Cfloat}, CuPtr{Cint}, CuPtr{Cint}, csrsv2Info_t, Ptr{Cint}),
                 handle, transA, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA,
                 csrSortedColIndA, info, pBufferSizeInBytes)
end

function cusparseDcsrsv2_bufferSize(handle, transA, m, nnz, descrA, csrSortedValA,
                                    csrSortedRowPtrA, csrSortedColIndA, info,
                                    pBufferSizeInBytes)
    @check @runtime_ccall((:cusparseDcsrsv2_bufferSize, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseOperation_t, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{Cdouble}, CuPtr{Cint}, CuPtr{Cint}, csrsv2Info_t, Ptr{Cint}),
                 handle, transA, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA,
                 csrSortedColIndA, info, pBufferSizeInBytes)
end

function cusparseCcsrsv2_bufferSize(handle, transA, m, nnz, descrA, csrSortedValA,
                                    csrSortedRowPtrA, csrSortedColIndA, info,
                                    pBufferSizeInBytes)
    @check @runtime_ccall((:cusparseCcsrsv2_bufferSize, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseOperation_t, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{cuComplex}, CuPtr{Cint}, CuPtr{Cint}, csrsv2Info_t, Ptr{Cint}),
                 handle, transA, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA,
                 csrSortedColIndA, info, pBufferSizeInBytes)
end

function cusparseZcsrsv2_bufferSize(handle, transA, m, nnz, descrA, csrSortedValA,
                                    csrSortedRowPtrA, csrSortedColIndA, info,
                                    pBufferSizeInBytes)
    @check @runtime_ccall((:cusparseZcsrsv2_bufferSize, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseOperation_t, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{cuDoubleComplex}, CuPtr{Cint}, CuPtr{Cint}, csrsv2Info_t, Ptr{Cint}),
                 handle, transA, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA,
                 csrSortedColIndA, info, pBufferSizeInBytes)
end

function cusparseScsrsv2_bufferSizeExt(handle, transA, m, nnz, descrA, csrSortedValA,
                                       csrSortedRowPtrA, csrSortedColIndA, info, pBufferSize)
    @check @runtime_ccall((:cusparseScsrsv2_bufferSizeExt, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseOperation_t, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{Cfloat}, CuPtr{Cint}, CuPtr{Cint}, csrsv2Info_t, Ptr{Csize_t}),
                 handle, transA, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA,
                 csrSortedColIndA, info, pBufferSize)
end

function cusparseDcsrsv2_bufferSizeExt(handle, transA, m, nnz, descrA, csrSortedValA,
                                       csrSortedRowPtrA, csrSortedColIndA, info, pBufferSize)
    @check @runtime_ccall((:cusparseDcsrsv2_bufferSizeExt, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseOperation_t, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{Cdouble}, CuPtr{Cint}, CuPtr{Cint}, csrsv2Info_t, Ptr{Csize_t}),
                 handle, transA, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA,
                 csrSortedColIndA, info, pBufferSize)
end

function cusparseCcsrsv2_bufferSizeExt(handle, transA, m, nnz, descrA, csrSortedValA,
                                       csrSortedRowPtrA, csrSortedColIndA, info, pBufferSize)
    @check @runtime_ccall((:cusparseCcsrsv2_bufferSizeExt, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseOperation_t, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{cuComplex}, CuPtr{Cint}, CuPtr{Cint}, csrsv2Info_t, Ptr{Csize_t}),
                 handle, transA, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA,
                 csrSortedColIndA, info, pBufferSize)
end

function cusparseZcsrsv2_bufferSizeExt(handle, transA, m, nnz, descrA, csrSortedValA,
                                       csrSortedRowPtrA, csrSortedColIndA, info, pBufferSize)
    @check @runtime_ccall((:cusparseZcsrsv2_bufferSizeExt, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseOperation_t, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{cuDoubleComplex}, CuPtr{Cint}, CuPtr{Cint}, csrsv2Info_t,
                  Ptr{Csize_t}),
                 handle, transA, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA,
                 csrSortedColIndA, info, pBufferSize)
end

function cusparseScsrsv2_analysis(handle, transA, m, nnz, descrA, csrSortedValA,
                                  csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer)
    @check @runtime_ccall((:cusparseScsrsv2_analysis, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseOperation_t, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{Cfloat}, CuPtr{Cint}, CuPtr{Cint}, csrsv2Info_t,
                  cusparseSolvePolicy_t, CuPtr{Cvoid}),
                 handle, transA, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA,
                 csrSortedColIndA, info, policy, pBuffer)
end

function cusparseDcsrsv2_analysis(handle, transA, m, nnz, descrA, csrSortedValA,
                                  csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer)
    @check @runtime_ccall((:cusparseDcsrsv2_analysis, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseOperation_t, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{Cdouble}, CuPtr{Cint}, CuPtr{Cint}, csrsv2Info_t,
                  cusparseSolvePolicy_t, CuPtr{Cvoid}),
                 handle, transA, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA,
                 csrSortedColIndA, info, policy, pBuffer)
end

function cusparseCcsrsv2_analysis(handle, transA, m, nnz, descrA, csrSortedValA,
                                  csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer)
    @check @runtime_ccall((:cusparseCcsrsv2_analysis, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseOperation_t, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{cuComplex}, CuPtr{Cint}, CuPtr{Cint}, csrsv2Info_t,
                  cusparseSolvePolicy_t, CuPtr{Cvoid}),
                 handle, transA, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA,
                 csrSortedColIndA, info, policy, pBuffer)
end

function cusparseZcsrsv2_analysis(handle, transA, m, nnz, descrA, csrSortedValA,
                                  csrSortedRowPtrA, csrSortedColIndA, info, policy, pBuffer)
    @check @runtime_ccall((:cusparseZcsrsv2_analysis, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseOperation_t, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{cuDoubleComplex}, CuPtr{Cint}, CuPtr{Cint}, csrsv2Info_t,
                  cusparseSolvePolicy_t, CuPtr{Cvoid}),
                 handle, transA, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA,
                 csrSortedColIndA, info, policy, pBuffer)
end

function cusparseScsrsv2_solve(handle, transA, m, nnz, alpha, descrA, csrSortedValA,
                               csrSortedRowPtrA, csrSortedColIndA, info, f, x, policy,
                               pBuffer)
    @check @runtime_ccall((:cusparseScsrsv2_solve, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseOperation_t, Cint, Cint, Ptr{Cfloat},
                  cusparseMatDescr_t, CuPtr{Cfloat}, CuPtr{Cint}, CuPtr{Cint},
                  csrsv2Info_t, CuPtr{Cfloat}, CuPtr{Cfloat}, cusparseSolvePolicy_t,
                  CuPtr{Cvoid}),
                 handle, transA, m, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA,
                 csrSortedColIndA, info, f, x, policy, pBuffer)
end

function cusparseDcsrsv2_solve(handle, transA, m, nnz, alpha, descrA, csrSortedValA,
                               csrSortedRowPtrA, csrSortedColIndA, info, f, x, policy,
                               pBuffer)
    @check @runtime_ccall((:cusparseDcsrsv2_solve, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseOperation_t, Cint, Cint, Ptr{Cdouble},
                  cusparseMatDescr_t, CuPtr{Cdouble}, CuPtr{Cint}, CuPtr{Cint},
                  csrsv2Info_t, CuPtr{Cdouble}, CuPtr{Cdouble}, cusparseSolvePolicy_t,
                  CuPtr{Cvoid}),
                 handle, transA, m, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA,
                 csrSortedColIndA, info, f, x, policy, pBuffer)
end

function cusparseCcsrsv2_solve(handle, transA, m, nnz, alpha, descrA, csrSortedValA,
                               csrSortedRowPtrA, csrSortedColIndA, info, f, x, policy,
                               pBuffer)
    @check @runtime_ccall((:cusparseCcsrsv2_solve, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseOperation_t, Cint, Cint, Ptr{cuComplex},
                  cusparseMatDescr_t, CuPtr{cuComplex}, CuPtr{Cint}, CuPtr{Cint},
                  csrsv2Info_t, CuPtr{cuComplex}, CuPtr{cuComplex}, cusparseSolvePolicy_t,
                  CuPtr{Cvoid}),
                 handle, transA, m, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA,
                 csrSortedColIndA, info, f, x, policy, pBuffer)
end

function cusparseZcsrsv2_solve(handle, transA, m, nnz, alpha, descrA, csrSortedValA,
                               csrSortedRowPtrA, csrSortedColIndA, info, f, x, policy,
                               pBuffer)
    @check @runtime_ccall((:cusparseZcsrsv2_solve, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseOperation_t, Cint, Cint, Ptr{cuDoubleComplex},
                  cusparseMatDescr_t, CuPtr{cuDoubleComplex}, CuPtr{Cint}, CuPtr{Cint},
                  csrsv2Info_t, CuPtr{cuDoubleComplex}, CuPtr{cuDoubleComplex},
                  cusparseSolvePolicy_t, CuPtr{Cvoid}),
                 handle, transA, m, nnz, alpha, descrA, csrSortedValA, csrSortedRowPtrA,
                 csrSortedColIndA, info, f, x, policy, pBuffer)
end

function cusparseXbsrsv2_zeroPivot(handle, info, position)
    @check @runtime_ccall((:cusparseXbsrsv2_zeroPivot, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, bsrsv2Info_t, Ptr{Cint}),
                 handle, info, position)
end

function cusparseSbsrsv2_bufferSize(handle, dirA, transA, mb, nnzb, descrA, bsrSortedValA,
                                    bsrSortedRowPtrA, bsrSortedColIndA, blockDim, info,
                                    pBufferSizeInBytes)
    @check @runtime_ccall((:cusparseSbsrsv2_bufferSize, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, Cint, Cint,
                  cusparseMatDescr_t, CuPtr{Cfloat}, CuPtr{Cint}, CuPtr{Cint}, Cint,
                  bsrsv2Info_t, Ptr{Cint}),
                 handle, dirA, transA, mb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA,
                 bsrSortedColIndA, blockDim, info, pBufferSizeInBytes)
end

function cusparseDbsrsv2_bufferSize(handle, dirA, transA, mb, nnzb, descrA, bsrSortedValA,
                                    bsrSortedRowPtrA, bsrSortedColIndA, blockDim, info,
                                    pBufferSizeInBytes)
    @check @runtime_ccall((:cusparseDbsrsv2_bufferSize, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, Cint, Cint,
                  cusparseMatDescr_t, CuPtr{Cdouble}, CuPtr{Cint}, CuPtr{Cint}, Cint,
                  bsrsv2Info_t, Ptr{Cint}),
                 handle, dirA, transA, mb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA,
                 bsrSortedColIndA, blockDim, info, pBufferSizeInBytes)
end

function cusparseCbsrsv2_bufferSize(handle, dirA, transA, mb, nnzb, descrA, bsrSortedValA,
                                    bsrSortedRowPtrA, bsrSortedColIndA, blockDim, info,
                                    pBufferSizeInBytes)
    @check @runtime_ccall((:cusparseCbsrsv2_bufferSize, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, Cint, Cint,
                  cusparseMatDescr_t, CuPtr{cuComplex}, CuPtr{Cint}, CuPtr{Cint}, Cint,
                  bsrsv2Info_t, Ptr{Cint}),
                 handle, dirA, transA, mb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA,
                 bsrSortedColIndA, blockDim, info, pBufferSizeInBytes)
end

function cusparseZbsrsv2_bufferSize(handle, dirA, transA, mb, nnzb, descrA, bsrSortedValA,
                                    bsrSortedRowPtrA, bsrSortedColIndA, blockDim, info,
                                    pBufferSizeInBytes)
    @check @runtime_ccall((:cusparseZbsrsv2_bufferSize, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, Cint, Cint,
                  cusparseMatDescr_t, CuPtr{cuDoubleComplex}, CuPtr{Cint}, CuPtr{Cint},
                  Cint, bsrsv2Info_t, Ptr{Cint}),
                 handle, dirA, transA, mb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA,
                 bsrSortedColIndA, blockDim, info, pBufferSizeInBytes)
end

function cusparseSbsrsv2_bufferSizeExt(handle, dirA, transA, mb, nnzb, descrA,
                                       bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA,
                                       blockSize, info, pBufferSize)
    @check @runtime_ccall((:cusparseSbsrsv2_bufferSizeExt, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, Cint, Cint,
                  cusparseMatDescr_t, CuPtr{Cfloat}, CuPtr{Cint}, CuPtr{Cint}, Cint,
                  bsrsv2Info_t, Ptr{Csize_t}),
                 handle, dirA, transA, mb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA,
                 bsrSortedColIndA, blockSize, info, pBufferSize)
end

function cusparseDbsrsv2_bufferSizeExt(handle, dirA, transA, mb, nnzb, descrA,
                                       bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA,
                                       blockSize, info, pBufferSize)
    @check @runtime_ccall((:cusparseDbsrsv2_bufferSizeExt, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, Cint, Cint,
                  cusparseMatDescr_t, CuPtr{Cdouble}, CuPtr{Cint}, CuPtr{Cint}, Cint,
                  bsrsv2Info_t, Ptr{Csize_t}),
                 handle, dirA, transA, mb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA,
                 bsrSortedColIndA, blockSize, info, pBufferSize)
end

function cusparseCbsrsv2_bufferSizeExt(handle, dirA, transA, mb, nnzb, descrA,
                                       bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA,
                                       blockSize, info, pBufferSize)
    @check @runtime_ccall((:cusparseCbsrsv2_bufferSizeExt, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, Cint, Cint,
                  cusparseMatDescr_t, CuPtr{cuComplex}, CuPtr{Cint}, CuPtr{Cint}, Cint,
                  bsrsv2Info_t, Ptr{Csize_t}),
                 handle, dirA, transA, mb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA,
                 bsrSortedColIndA, blockSize, info, pBufferSize)
end

function cusparseZbsrsv2_bufferSizeExt(handle, dirA, transA, mb, nnzb, descrA,
                                       bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA,
                                       blockSize, info, pBufferSize)
    @check @runtime_ccall((:cusparseZbsrsv2_bufferSizeExt, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, Cint, Cint,
                  cusparseMatDescr_t, CuPtr{cuDoubleComplex}, CuPtr{Cint}, CuPtr{Cint},
                  Cint, bsrsv2Info_t, Ptr{Csize_t}),
                 handle, dirA, transA, mb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA,
                 bsrSortedColIndA, blockSize, info, pBufferSize)
end

function cusparseSbsrsv2_analysis(handle, dirA, transA, mb, nnzb, descrA, bsrSortedValA,
                                  bsrSortedRowPtrA, bsrSortedColIndA, blockDim, info,
                                  policy, pBuffer)
    @check @runtime_ccall((:cusparseSbsrsv2_analysis, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, Cint, Cint,
                  cusparseMatDescr_t, CuPtr{Cfloat}, CuPtr{Cint}, CuPtr{Cint}, Cint,
                  bsrsv2Info_t, cusparseSolvePolicy_t, CuPtr{Cvoid}),
                 handle, dirA, transA, mb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA,
                 bsrSortedColIndA, blockDim, info, policy, pBuffer)
end

function cusparseDbsrsv2_analysis(handle, dirA, transA, mb, nnzb, descrA, bsrSortedValA,
                                  bsrSortedRowPtrA, bsrSortedColIndA, blockDim, info,
                                  policy, pBuffer)
    @check @runtime_ccall((:cusparseDbsrsv2_analysis, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, Cint, Cint,
                  cusparseMatDescr_t, CuPtr{Cdouble}, CuPtr{Cint}, CuPtr{Cint}, Cint,
                  bsrsv2Info_t, cusparseSolvePolicy_t, CuPtr{Cvoid}),
                 handle, dirA, transA, mb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA,
                 bsrSortedColIndA, blockDim, info, policy, pBuffer)
end

function cusparseCbsrsv2_analysis(handle, dirA, transA, mb, nnzb, descrA, bsrSortedValA,
                                  bsrSortedRowPtrA, bsrSortedColIndA, blockDim, info,
                                  policy, pBuffer)
    @check @runtime_ccall((:cusparseCbsrsv2_analysis, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, Cint, Cint,
                  cusparseMatDescr_t, CuPtr{cuComplex}, CuPtr{Cint}, CuPtr{Cint}, Cint,
                  bsrsv2Info_t, cusparseSolvePolicy_t, CuPtr{Cvoid}),
                 handle, dirA, transA, mb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA,
                 bsrSortedColIndA, blockDim, info, policy, pBuffer)
end

function cusparseZbsrsv2_analysis(handle, dirA, transA, mb, nnzb, descrA, bsrSortedValA,
                                  bsrSortedRowPtrA, bsrSortedColIndA, blockDim, info,
                                  policy, pBuffer)
    @check @runtime_ccall((:cusparseZbsrsv2_analysis, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, Cint, Cint,
                  cusparseMatDescr_t, CuPtr{cuDoubleComplex}, CuPtr{Cint}, CuPtr{Cint},
                  Cint, bsrsv2Info_t, cusparseSolvePolicy_t, CuPtr{Cvoid}),
                 handle, dirA, transA, mb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA,
                 bsrSortedColIndA, blockDim, info, policy, pBuffer)
end

function cusparseSbsrsv2_solve(handle, dirA, transA, mb, nnzb, alpha, descrA,
                               bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockDim,
                               info, f, x, policy, pBuffer)
    @check @runtime_ccall((:cusparseSbsrsv2_solve, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, Cint, Cint,
                  Ptr{Cfloat}, cusparseMatDescr_t, CuPtr{Cfloat}, CuPtr{Cint}, CuPtr{Cint},
                  Cint, bsrsv2Info_t, CuPtr{Cfloat}, CuPtr{Cfloat}, cusparseSolvePolicy_t,
                  CuPtr{Cvoid}),
                 handle, dirA, transA, mb, nnzb, alpha, descrA, bsrSortedValA,
                 bsrSortedRowPtrA, bsrSortedColIndA, blockDim, info, f, x, policy, pBuffer)
end

function cusparseDbsrsv2_solve(handle, dirA, transA, mb, nnzb, alpha, descrA,
                               bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockDim,
                               info, f, x, policy, pBuffer)
    @check @runtime_ccall((:cusparseDbsrsv2_solve, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, Cint, Cint,
                  Ptr{Cdouble}, cusparseMatDescr_t, CuPtr{Cdouble}, CuPtr{Cint},
                  CuPtr{Cint}, Cint, bsrsv2Info_t, CuPtr{Cdouble}, CuPtr{Cdouble},
                  cusparseSolvePolicy_t, CuPtr{Cvoid}),
                 handle, dirA, transA, mb, nnzb, alpha, descrA, bsrSortedValA,
                 bsrSortedRowPtrA, bsrSortedColIndA, blockDim, info, f, x, policy, pBuffer)
end

function cusparseCbsrsv2_solve(handle, dirA, transA, mb, nnzb, alpha, descrA,
                               bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockDim,
                               info, f, x, policy, pBuffer)
    @check @runtime_ccall((:cusparseCbsrsv2_solve, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, Cint, Cint,
                  Ptr{cuComplex}, cusparseMatDescr_t, CuPtr{cuComplex}, CuPtr{Cint},
                  CuPtr{Cint}, Cint, bsrsv2Info_t, CuPtr{cuComplex}, CuPtr{cuComplex},
                  cusparseSolvePolicy_t, CuPtr{Cvoid}),
                 handle, dirA, transA, mb, nnzb, alpha, descrA, bsrSortedValA,
                 bsrSortedRowPtrA, bsrSortedColIndA, blockDim, info, f, x, policy, pBuffer)
end

function cusparseZbsrsv2_solve(handle, dirA, transA, mb, nnzb, alpha, descrA,
                               bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockDim,
                               info, f, x, policy, pBuffer)
    @check @runtime_ccall((:cusparseZbsrsv2_solve, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, cusparseOperation_t, Cint, Cint,
                  Ptr{cuDoubleComplex}, cusparseMatDescr_t, CuPtr{cuDoubleComplex},
                  CuPtr{Cint}, CuPtr{Cint}, Cint, bsrsv2Info_t, CuPtr{cuDoubleComplex},
                  CuPtr{cuDoubleComplex}, cusparseSolvePolicy_t, CuPtr{Cvoid}),
                 handle, dirA, transA, mb, nnzb, alpha, descrA, bsrSortedValA,
                 bsrSortedRowPtrA, bsrSortedColIndA, blockDim, info, f, x, policy, pBuffer)
end

function cusparseShybsv_analysis(handle, transA, descrA, hybA, info)
    @check @runtime_ccall((:cusparseShybsv_analysis, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseOperation_t, cusparseMatDescr_t,
                  cusparseHybMat_t, cusparseSolveAnalysisInfo_t),
                 handle, transA, descrA, hybA, info)
end

function cusparseDhybsv_analysis(handle, transA, descrA, hybA, info)
    @check @runtime_ccall((:cusparseDhybsv_analysis, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseOperation_t, cusparseMatDescr_t,
                  cusparseHybMat_t, cusparseSolveAnalysisInfo_t),
                 handle, transA, descrA, hybA, info)
end

function cusparseChybsv_analysis(handle, transA, descrA, hybA, info)
    @check @runtime_ccall((:cusparseChybsv_analysis, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseOperation_t, cusparseMatDescr_t,
                  cusparseHybMat_t, cusparseSolveAnalysisInfo_t),
                 handle, transA, descrA, hybA, info)
end

function cusparseZhybsv_analysis(handle, transA, descrA, hybA, info)
    @check @runtime_ccall((:cusparseZhybsv_analysis, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseOperation_t, cusparseMatDescr_t,
                  cusparseHybMat_t, cusparseSolveAnalysisInfo_t),
                 handle, transA, descrA, hybA, info)
end

function cusparseShybsv_solve(handle, trans, alpha, descrA, hybA, info, f, x)
    @check @runtime_ccall((:cusparseShybsv_solve, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseOperation_t, Ptr{Cfloat}, cusparseMatDescr_t,
                  cusparseHybMat_t, cusparseSolveAnalysisInfo_t, CuPtr{Cfloat},
                  CuPtr{Cfloat}),
                 handle, trans, alpha, descrA, hybA, info, f, x)
end

function cusparseChybsv_solve(handle, trans, alpha, descrA, hybA, info, f, x)
    @check @runtime_ccall((:cusparseChybsv_solve, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseOperation_t, Ptr{cuComplex},
                  cusparseMatDescr_t, cusparseHybMat_t, cusparseSolveAnalysisInfo_t,
                  CuPtr{cuComplex}, CuPtr{cuComplex}),
                 handle, trans, alpha, descrA, hybA, info, f, x)
end

function cusparseDhybsv_solve(handle, trans, alpha, descrA, hybA, info, f, x)
    @check @runtime_ccall((:cusparseDhybsv_solve, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseOperation_t, Ptr{Cdouble}, cusparseMatDescr_t,
                  cusparseHybMat_t, cusparseSolveAnalysisInfo_t, CuPtr{Cdouble},
                  CuPtr{Cdouble}),
                 handle, trans, alpha, descrA, hybA, info, f, x)
end

function cusparseZhybsv_solve(handle, trans, alpha, descrA, hybA, info, f, x)
    @check @runtime_ccall((:cusparseZhybsv_solve, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseOperation_t, Ptr{cuDoubleComplex},
                  cusparseMatDescr_t, cusparseHybMat_t, cusparseSolveAnalysisInfo_t,
                  CuPtr{cuDoubleComplex}, CuPtr{cuDoubleComplex}),
                 handle, trans, alpha, descrA, hybA, info, f, x)
end

function cusparseScsrmm(handle, transA, m, n, k, nnz, alpha, descrA, csrSortedValA,
                        csrSortedRowPtrA, csrSortedColIndA, B, ldb, beta, C, ldc)
    @check @runtime_ccall((:cusparseScsrmm, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseOperation_t, Cint, Cint, Cint, Cint,
                  Ptr{Cfloat}, cusparseMatDescr_t, CuPtr{Cfloat}, CuPtr{Cint}, CuPtr{Cint},
                  CuPtr{Cfloat}, Cint, Ptr{Cfloat}, CuPtr{Cfloat}, Cint),
                 handle, transA, m, n, k, nnz, alpha, descrA, csrSortedValA,
                 csrSortedRowPtrA, csrSortedColIndA, B, ldb, beta, C, ldc)
end

function cusparseDcsrmm(handle, transA, m, n, k, nnz, alpha, descrA, csrSortedValA,
                        csrSortedRowPtrA, csrSortedColIndA, B, ldb, beta, C, ldc)
    @check @runtime_ccall((:cusparseDcsrmm, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseOperation_t, Cint, Cint, Cint, Cint,
                  Ptr{Cdouble}, cusparseMatDescr_t, CuPtr{Cdouble}, CuPtr{Cint},
                  CuPtr{Cint}, CuPtr{Cdouble}, Cint, Ptr{Cdouble}, CuPtr{Cdouble}, Cint),
                 handle, transA, m, n, k, nnz, alpha, descrA, csrSortedValA,
                 csrSortedRowPtrA, csrSortedColIndA, B, ldb, beta, C, ldc)
end

function cusparseCcsrmm(handle, transA, m, n, k, nnz, alpha, descrA, csrSortedValA,
                        csrSortedRowPtrA, csrSortedColIndA, B, ldb, beta, C, ldc)
    @check @runtime_ccall((:cusparseCcsrmm, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseOperation_t, Cint, Cint, Cint, Cint,
                  Ptr{cuComplex}, cusparseMatDescr_t, CuPtr{cuComplex}, CuPtr{Cint},
                  CuPtr{Cint}, CuPtr{cuComplex}, Cint, Ptr{cuComplex}, CuPtr{cuComplex},
                  Cint),
                 handle, transA, m, n, k, nnz, alpha, descrA, csrSortedValA,
                 csrSortedRowPtrA, csrSortedColIndA, B, ldb, beta, C, ldc)
end

function cusparseZcsrmm(handle, transA, m, n, k, nnz, alpha, descrA, csrSortedValA,
                        csrSortedRowPtrA, csrSortedColIndA, B, ldb, beta, C, ldc)
    @check @runtime_ccall((:cusparseZcsrmm, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseOperation_t, Cint, Cint, Cint, Cint,
                  Ptr{cuDoubleComplex}, cusparseMatDescr_t, CuPtr{cuDoubleComplex},
                  CuPtr{Cint}, CuPtr{Cint}, CuPtr{cuDoubleComplex}, Cint,
                  Ptr{cuDoubleComplex}, CuPtr{cuDoubleComplex}, Cint),
                 handle, transA, m, n, k, nnz, alpha, descrA, csrSortedValA,
                 csrSortedRowPtrA, csrSortedColIndA, B, ldb, beta, C, ldc)
end

function cusparseScsrmm2(handle, transA, transB, m, n, k, nnz, alpha, descrA,
                         csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, B, ldb, beta,
                         C, ldc)
    @check @runtime_ccall((:cusparseScsrmm2, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, Cint, Cint,
                  Cint, Cint, Ptr{Cfloat}, cusparseMatDescr_t, CuPtr{Cfloat}, CuPtr{Cint},
                  CuPtr{Cint}, CuPtr{Cfloat}, Cint, Ptr{Cfloat}, CuPtr{Cfloat}, Cint),
                 handle, transA, transB, m, n, k, nnz, alpha, descrA, csrSortedValA,
                 csrSortedRowPtrA, csrSortedColIndA, B, ldb, beta, C, ldc)
end

function cusparseDcsrmm2(handle, transA, transB, m, n, k, nnz, alpha, descrA,
                         csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, B, ldb, beta,
                         C, ldc)
    @check @runtime_ccall((:cusparseDcsrmm2, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, Cint, Cint,
                  Cint, Cint, Ptr{Cdouble}, cusparseMatDescr_t, CuPtr{Cdouble},
                  CuPtr{Cint}, CuPtr{Cint}, CuPtr{Cdouble}, Cint, Ptr{Cdouble},
                  CuPtr{Cdouble}, Cint),
                 handle, transA, transB, m, n, k, nnz, alpha, descrA, csrSortedValA,
                 csrSortedRowPtrA, csrSortedColIndA, B, ldb, beta, C, ldc)
end

function cusparseCcsrmm2(handle, transA, transB, m, n, k, nnz, alpha, descrA,
                         csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, B, ldb, beta,
                         C, ldc)
    @check @runtime_ccall((:cusparseCcsrmm2, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, Cint, Cint,
                  Cint, Cint, Ptr{cuComplex}, cusparseMatDescr_t, CuPtr{cuComplex},
                  CuPtr{Cint}, CuPtr{Cint}, CuPtr{cuComplex}, Cint, Ptr{cuComplex},
                  CuPtr{cuComplex}, Cint),
                 handle, transA, transB, m, n, k, nnz, alpha, descrA, csrSortedValA,
                 csrSortedRowPtrA, csrSortedColIndA, B, ldb, beta, C, ldc)
end

function cusparseZcsrmm2(handle, transA, transB, m, n, k, nnz, alpha, descrA,
                         csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, B, ldb, beta,
                         C, ldc)
    @check @runtime_ccall((:cusparseZcsrmm2, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, Cint, Cint,
                  Cint, Cint, Ptr{cuDoubleComplex}, cusparseMatDescr_t,
                  CuPtr{cuDoubleComplex}, CuPtr{Cint}, CuPtr{Cint}, CuPtr{cuDoubleComplex},
                  Cint, Ptr{cuDoubleComplex}, CuPtr{cuDoubleComplex}, Cint),
                 handle, transA, transB, m, n, k, nnz, alpha, descrA, csrSortedValA,
                 csrSortedRowPtrA, csrSortedColIndA, B, ldb, beta, C, ldc)
end

function cusparseSbsrmm(handle, dirA, transA, transB, mb, n, kb, nnzb, alpha, descrA,
                        bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockSize, B,
                        ldb, beta, C, ldc)
    @check @runtime_ccall((:cusparseSbsrmm, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, cusparseOperation_t,
                  cusparseOperation_t, Cint, Cint, Cint, Cint, Ptr{Cfloat},
                  cusparseMatDescr_t, CuPtr{Cfloat}, CuPtr{Cint}, CuPtr{Cint}, Cint,
                  CuPtr{Cfloat}, Cint, Ptr{Cfloat}, CuPtr{Cfloat}, Cint),
                 handle, dirA, transA, transB, mb, n, kb, nnzb, alpha, descrA,
                 bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockSize, B, ldb,
                 beta, C, ldc)
end

function cusparseDbsrmm(handle, dirA, transA, transB, mb, n, kb, nnzb, alpha, descrA,
                        bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockSize, B,
                        ldb, beta, C, ldc)
    @check @runtime_ccall((:cusparseDbsrmm, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, cusparseOperation_t,
                  cusparseOperation_t, Cint, Cint, Cint, Cint, Ptr{Cdouble},
                  cusparseMatDescr_t, CuPtr{Cdouble}, CuPtr{Cint}, CuPtr{Cint}, Cint,
                  CuPtr{Cdouble}, Cint, Ptr{Cdouble}, CuPtr{Cdouble}, Cint),
                 handle, dirA, transA, transB, mb, n, kb, nnzb, alpha, descrA,
                 bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockSize, B, ldb,
                 beta, C, ldc)
end

function cusparseCbsrmm(handle, dirA, transA, transB, mb, n, kb, nnzb, alpha, descrA,
                        bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockSize, B,
                        ldb, beta, C, ldc)
    @check @runtime_ccall((:cusparseCbsrmm, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, cusparseOperation_t,
                  cusparseOperation_t, Cint, Cint, Cint, Cint, Ptr{cuComplex},
                  cusparseMatDescr_t, CuPtr{cuComplex}, CuPtr{Cint}, CuPtr{Cint}, Cint,
                  CuPtr{cuComplex}, Cint, Ptr{cuComplex}, CuPtr{cuComplex}, Cint),
                 handle, dirA, transA, transB, mb, n, kb, nnzb, alpha, descrA,
                 bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockSize, B, ldb,
                 beta, C, ldc)
end

function cusparseZbsrmm(handle, dirA, transA, transB, mb, n, kb, nnzb, alpha, descrA,
                        bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockSize, B,
                        ldb, beta, C, ldc)
    @check @runtime_ccall((:cusparseZbsrmm, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, cusparseOperation_t,
                  cusparseOperation_t, Cint, Cint, Cint, Cint, Ptr{cuDoubleComplex},
                  cusparseMatDescr_t, CuPtr{cuDoubleComplex}, CuPtr{Cint}, CuPtr{Cint},
                  Cint, CuPtr{cuDoubleComplex}, Cint, Ptr{cuDoubleComplex},
                  CuPtr{cuDoubleComplex}, Cint),
                 handle, dirA, transA, transB, mb, n, kb, nnzb, alpha, descrA,
                 bsrSortedValA, bsrSortedRowPtrA, bsrSortedColIndA, blockSize, B, ldb,
                 beta, C, ldc)
end

function cusparseSgemmi(handle, m, n, k, nnz, alpha, A, lda, cscValB, cscColPtrB,
                        cscRowIndB, beta, C, ldc)
    @check @runtime_ccall((:cusparseSgemmi, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, Cint, Cint, Ptr{Cfloat}, CuPtr{Cfloat},
                  Cint, CuPtr{Cfloat}, CuPtr{Cint}, CuPtr{Cint}, Ptr{Cfloat},
                  CuPtr{Cfloat}, Cint),
                 handle, m, n, k, nnz, alpha, A, lda, cscValB, cscColPtrB, cscRowIndB,
                 beta, C, ldc)
end

function cusparseDgemmi(handle, m, n, k, nnz, alpha, A, lda, cscValB, cscColPtrB,
                        cscRowIndB, beta, C, ldc)
    @check @runtime_ccall((:cusparseDgemmi, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, Cint, Cint, Ptr{Cdouble}, CuPtr{Cdouble},
                  Cint, CuPtr{Cdouble}, CuPtr{Cint}, CuPtr{Cint}, Ptr{Cdouble},
                  CuPtr{Cdouble}, Cint),
                 handle, m, n, k, nnz, alpha, A, lda, cscValB, cscColPtrB, cscRowIndB,
                 beta, C, ldc)
end

function cusparseCgemmi(handle, m, n, k, nnz, alpha, A, lda, cscValB, cscColPtrB,
                        cscRowIndB, beta, C, ldc)
    @check @runtime_ccall((:cusparseCgemmi, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, Cint, Cint, Ptr{cuComplex},
                  CuPtr{cuComplex}, Cint, CuPtr{cuComplex}, CuPtr{Cint}, CuPtr{Cint},
                  Ptr{cuComplex}, CuPtr{cuComplex}, Cint),
                 handle, m, n, k, nnz, alpha, A, lda, cscValB, cscColPtrB, cscRowIndB,
                 beta, C, ldc)
end

function cusparseZgemmi(handle, m, n, k, nnz, alpha, A, lda, cscValB, cscColPtrB,
                        cscRowIndB, beta, C, ldc)
    @check @runtime_ccall((:cusparseZgemmi, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, Cint, Cint, Ptr{cuDoubleComplex},
                  CuPtr{cuDoubleComplex}, Cint, CuPtr{cuDoubleComplex}, CuPtr{Cint},
                  CuPtr{Cint}, Ptr{cuDoubleComplex}, CuPtr{cuDoubleComplex}, Cint),
                 handle, m, n, k, nnz, alpha, A, lda, cscValB, cscColPtrB, cscRowIndB,
                 beta, C, ldc)
end

function cusparseScsrsm_analysis(handle, transA, m, nnz, descrA, csrSortedValA,
                                 csrSortedRowPtrA, csrSortedColIndA, info)
    @check @runtime_ccall((:cusparseScsrsm_analysis, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseOperation_t, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{Cfloat}, CuPtr{Cint}, CuPtr{Cint}, cusparseSolveAnalysisInfo_t),
                 handle, transA, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA,
                 csrSortedColIndA, info)
end

function cusparseDcsrsm_analysis(handle, transA, m, nnz, descrA, csrSortedValA,
                                 csrSortedRowPtrA, csrSortedColIndA, info)
    @check @runtime_ccall((:cusparseDcsrsm_analysis, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseOperation_t, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{Cdouble}, CuPtr{Cint}, CuPtr{Cint}, cusparseSolveAnalysisInfo_t),
                 handle, transA, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA,
                 csrSortedColIndA, info)
end

function cusparseCcsrsm_analysis(handle, transA, m, nnz, descrA, csrSortedValA,
                                 csrSortedRowPtrA, csrSortedColIndA, info)
    @check @runtime_ccall((:cusparseCcsrsm_analysis, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseOperation_t, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{cuComplex}, CuPtr{Cint}, CuPtr{Cint}, cusparseSolveAnalysisInfo_t),
                 handle, transA, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA,
                 csrSortedColIndA, info)
end

function cusparseZcsrsm_analysis(handle, transA, m, nnz, descrA, csrSortedValA,
                                 csrSortedRowPtrA, csrSortedColIndA, info)
    @check @runtime_ccall((:cusparseZcsrsm_analysis, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseOperation_t, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{cuDoubleComplex}, CuPtr{Cint}, CuPtr{Cint},
                  cusparseSolveAnalysisInfo_t),
                 handle, transA, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA,
                 csrSortedColIndA, info)
end

function cusparseScsrsm_solve(handle, transA, m, n, alpha, descrA, csrSortedValA,
                              csrSortedRowPtrA, csrSortedColIndA, info, B, ldb, X, ldx)
    @check @runtime_ccall((:cusparseScsrsm_solve, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseOperation_t, Cint, Cint, Ptr{Cfloat},
                  cusparseMatDescr_t, CuPtr{Cfloat}, CuPtr{Cint}, CuPtr{Cint},
                  cusparseSolveAnalysisInfo_t, CuPtr{Cfloat}, Cint, CuPtr{Cfloat}, Cint),
                 handle, transA, m, n, alpha, descrA, csrSortedValA, csrSortedRowPtrA,
                 csrSortedColIndA, info, B, ldb, X, ldx)
end

function cusparseDcsrsm_solve(handle, transA, m, n, alpha, descrA, csrSortedValA,
                              csrSortedRowPtrA, csrSortedColIndA, info, B, ldb, X, ldx)
    @check @runtime_ccall((:cusparseDcsrsm_solve, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseOperation_t, Cint, Cint, Ptr{Cdouble},
                  cusparseMatDescr_t, CuPtr{Cdouble}, CuPtr{Cint}, CuPtr{Cint},
                  cusparseSolveAnalysisInfo_t, CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, Cint),
                 handle, transA, m, n, alpha, descrA, csrSortedValA, csrSortedRowPtrA,
                 csrSortedColIndA, info, B, ldb, X, ldx)
end

function cusparseCcsrsm_solve(handle, transA, m, n, alpha, descrA, csrSortedValA,
                              csrSortedRowPtrA, csrSortedColIndA, info, B, ldb, X, ldx)
    @check @runtime_ccall((:cusparseCcsrsm_solve, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseOperation_t, Cint, Cint, Ptr{cuComplex},
                  cusparseMatDescr_t, CuPtr{cuComplex}, CuPtr{Cint}, CuPtr{Cint},
                  cusparseSolveAnalysisInfo_t, CuPtr{cuComplex}, Cint, CuPtr{cuComplex},
                  Cint),
                 handle, transA, m, n, alpha, descrA, csrSortedValA, csrSortedRowPtrA,
                 csrSortedColIndA, info, B, ldb, X, ldx)
end

function cusparseZcsrsm_solve(handle, transA, m, n, alpha, descrA, csrSortedValA,
                              csrSortedRowPtrA, csrSortedColIndA, info, B, ldb, X, ldx)
    @check @runtime_ccall((:cusparseZcsrsm_solve, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseOperation_t, Cint, Cint, Ptr{cuDoubleComplex},
                  cusparseMatDescr_t, CuPtr{cuDoubleComplex}, CuPtr{Cint}, CuPtr{Cint},
                  cusparseSolveAnalysisInfo_t, CuPtr{cuDoubleComplex}, Cint,
                  CuPtr{cuDoubleComplex}, Cint),
                 handle, transA, m, n, alpha, descrA, csrSortedValA, csrSortedRowPtrA,
                 csrSortedColIndA, info, B, ldb, X, ldx)
end

function cusparseCreateCsrsm2Info(info)
    @check @runtime_ccall((:cusparseCreateCsrsm2Info, libcusparse()), cusparseStatus_t,
                 (Ptr{csrsm2Info_t},),
                 info)
end

function cusparseDestroyCsrsm2Info(info)
    @check @runtime_ccall((:cusparseDestroyCsrsm2Info, libcusparse()), cusparseStatus_t,
                 (csrsm2Info_t,),
                 info)
end

function cusparseXcsrsm2_zeroPivot(handle, info, position)
    @check @runtime_ccall((:cusparseXcsrsm2_zeroPivot, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, csrsm2Info_t, Ptr{Cint}),
                 handle, info, position)
end

function cusparseScsrsm2_bufferSizeExt(handle, algo, transA, transB, m, nrhs, nnz, alpha,
                                       descrA, csrSortedValA, csrSortedRowPtrA,
                                       csrSortedColIndA, B, ldb, info, policy, pBufferSize)
    @check @runtime_ccall((:cusparseScsrsm2_bufferSizeExt, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, cusparseOperation_t, cusparseOperation_t, Cint,
                  Cint, Cint, Ptr{Cfloat}, cusparseMatDescr_t, CuPtr{Cfloat}, CuPtr{Cint},
                  CuPtr{Cint}, CuPtr{Cfloat}, Cint, csrsm2Info_t, cusparseSolvePolicy_t,
                  Ptr{Csize_t}),
                 handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrSortedValA,
                 csrSortedRowPtrA, csrSortedColIndA, B, ldb, info, policy, pBufferSize)
end

function cusparseDcsrsm2_bufferSizeExt(handle, algo, transA, transB, m, nrhs, nnz, alpha,
                                       descrA, csrSortedValA, csrSortedRowPtrA,
                                       csrSortedColIndA, B, ldb, info, policy, pBufferSize)
    @check @runtime_ccall((:cusparseDcsrsm2_bufferSizeExt, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, cusparseOperation_t, cusparseOperation_t, Cint,
                  Cint, Cint, Ptr{Cdouble}, cusparseMatDescr_t, CuPtr{Cdouble},
                  CuPtr{Cint}, CuPtr{Cint}, CuPtr{Cdouble}, Cint, csrsm2Info_t,
                  cusparseSolvePolicy_t, Ptr{Csize_t}),
                 handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrSortedValA,
                 csrSortedRowPtrA, csrSortedColIndA, B, ldb, info, policy, pBufferSize)
end

function cusparseCcsrsm2_bufferSizeExt(handle, algo, transA, transB, m, nrhs, nnz, alpha,
                                       descrA, csrSortedValA, csrSortedRowPtrA,
                                       csrSortedColIndA, B, ldb, info, policy, pBufferSize)
    @check @runtime_ccall((:cusparseCcsrsm2_bufferSizeExt, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, cusparseOperation_t, cusparseOperation_t, Cint,
                  Cint, Cint, Ptr{cuComplex}, cusparseMatDescr_t, CuPtr{cuComplex},
                  CuPtr{Cint}, CuPtr{Cint}, CuPtr{cuComplex}, Cint, csrsm2Info_t,
                  cusparseSolvePolicy_t, Ptr{Csize_t}),
                 handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrSortedValA,
                 csrSortedRowPtrA, csrSortedColIndA, B, ldb, info, policy, pBufferSize)
end

function cusparseZcsrsm2_bufferSizeExt(handle, algo, transA, transB, m, nrhs, nnz, alpha,
                                       descrA, csrSortedValA, csrSortedRowPtrA,
                                       csrSortedColIndA, B, ldb, info, policy, pBufferSize)
    @check @runtime_ccall((:cusparseZcsrsm2_bufferSizeExt, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, cusparseOperation_t, cusparseOperation_t, Cint,
                  Cint, Cint, Ptr{cuDoubleComplex}, cusparseMatDescr_t,
                  CuPtr{cuDoubleComplex}, CuPtr{Cint}, CuPtr{Cint}, CuPtr{cuDoubleComplex},
                  Cint, csrsm2Info_t, cusparseSolvePolicy_t, Ptr{Csize_t}),
                 handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrSortedValA,
                 csrSortedRowPtrA, csrSortedColIndA, B, ldb, info, policy, pBufferSize)
end

function cusparseScsrsm2_analysis(handle, algo, transA, transB, m, nrhs, nnz, alpha,
                                  descrA, csrSortedValA, csrSortedRowPtrA,
                                  csrSortedColIndA, B, ldb, info, policy, pBuffer)
    @check @runtime_ccall((:cusparseScsrsm2_analysis, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, cusparseOperation_t, cusparseOperation_t, Cint,
                  Cint, Cint, Ptr{Cfloat}, cusparseMatDescr_t, CuPtr{Cfloat}, CuPtr{Cint},
                  CuPtr{Cint}, CuPtr{Cfloat}, Cint, csrsm2Info_t, cusparseSolvePolicy_t,
                  CuPtr{Cvoid}),
                 handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrSortedValA,
                 csrSortedRowPtrA, csrSortedColIndA, B, ldb, info, policy, pBuffer)
end

function cusparseDcsrsm2_analysis(handle, algo, transA, transB, m, nrhs, nnz, alpha,
                                  descrA, csrSortedValA, csrSortedRowPtrA,
                                  csrSortedColIndA, B, ldb, info, policy, pBuffer)
    @check @runtime_ccall((:cusparseDcsrsm2_analysis, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, cusparseOperation_t, cusparseOperation_t, Cint,
                  Cint, Cint, Ptr{Cdouble}, cusparseMatDescr_t, CuPtr{Cdouble},
                  CuPtr{Cint}, CuPtr{Cint}, CuPtr{Cdouble}, Cint, csrsm2Info_t,
                  cusparseSolvePolicy_t, CuPtr{Cvoid}),
                 handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrSortedValA,
                 csrSortedRowPtrA, csrSortedColIndA, B, ldb, info, policy, pBuffer)
end

function cusparseCcsrsm2_analysis(handle, algo, transA, transB, m, nrhs, nnz, alpha,
                                  descrA, csrSortedValA, csrSortedRowPtrA,
                                  csrSortedColIndA, B, ldb, info, policy, pBuffer)
    @check @runtime_ccall((:cusparseCcsrsm2_analysis, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, cusparseOperation_t, cusparseOperation_t, Cint,
                  Cint, Cint, Ptr{cuComplex}, cusparseMatDescr_t, CuPtr{cuComplex},
                  CuPtr{Cint}, CuPtr{Cint}, CuPtr{cuComplex}, Cint, csrsm2Info_t,
                  cusparseSolvePolicy_t, CuPtr{Cvoid}),
                 handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrSortedValA,
                 csrSortedRowPtrA, csrSortedColIndA, B, ldb, info, policy, pBuffer)
end

function cusparseZcsrsm2_analysis(handle, algo, transA, transB, m, nrhs, nnz, alpha,
                                  descrA, csrSortedValA, csrSortedRowPtrA,
                                  csrSortedColIndA, B, ldb, info, policy, pBuffer)
    @check @runtime_ccall((:cusparseZcsrsm2_analysis, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, cusparseOperation_t, cusparseOperation_t, Cint,
                  Cint, Cint, Ptr{cuDoubleComplex}, cusparseMatDescr_t,
                  CuPtr{cuDoubleComplex}, CuPtr{Cint}, CuPtr{Cint}, CuPtr{cuDoubleComplex},
                  Cint, csrsm2Info_t, cusparseSolvePolicy_t, CuPtr{Cvoid}),
                 handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrSortedValA,
                 csrSortedRowPtrA, csrSortedColIndA, B, ldb, info, policy, pBuffer)
end

function cusparseScsrsm2_solve(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA,
                               csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, B, ldb,
                               info, policy, pBuffer)
    @check @runtime_ccall((:cusparseScsrsm2_solve, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, cusparseOperation_t, cusparseOperation_t, Cint,
                  Cint, Cint, Ptr{Cfloat}, cusparseMatDescr_t, CuPtr{Cfloat}, CuPtr{Cint},
                  CuPtr{Cint}, CuPtr{Cfloat}, Cint, csrsm2Info_t, cusparseSolvePolicy_t,
                  CuPtr{Cvoid}),
                 handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrSortedValA,
                 csrSortedRowPtrA, csrSortedColIndA, B, ldb, info, policy, pBuffer)
end

function cusparseDcsrsm2_solve(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA,
                               csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, B, ldb,
                               info, policy, pBuffer)
    @check @runtime_ccall((:cusparseDcsrsm2_solve, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, cusparseOperation_t, cusparseOperation_t, Cint,
                  Cint, Cint, Ptr{Cdouble}, cusparseMatDescr_t, CuPtr{Cdouble},
                  CuPtr{Cint}, CuPtr{Cint}, CuPtr{Cdouble}, Cint, csrsm2Info_t,
                  cusparseSolvePolicy_t, CuPtr{Cvoid}),
                 handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrSortedValA,
                 csrSortedRowPtrA, csrSortedColIndA, B, ldb, info, policy, pBuffer)
end

function cusparseCcsrsm2_solve(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA,
                               csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, B, ldb,
                               info, policy, pBuffer)
    @check @runtime_ccall((:cusparseCcsrsm2_solve, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, cusparseOperation_t, cusparseOperation_t, Cint,
                  Cint, Cint, Ptr{cuComplex}, cusparseMatDescr_t, CuPtr{cuComplex},
                  CuPtr{Cint}, CuPtr{Cint}, CuPtr{cuComplex}, Cint, csrsm2Info_t,
                  cusparseSolvePolicy_t, CuPtr{Cvoid}),
                 handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrSortedValA,
                 csrSortedRowPtrA, csrSortedColIndA, B, ldb, info, policy, pBuffer)
end

function cusparseZcsrsm2_solve(handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA,
                               csrSortedValA, csrSortedRowPtrA, csrSortedColIndA, B, ldb,
                               info, policy, pBuffer)
    @check @runtime_ccall((:cusparseZcsrsm2_solve, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, cusparseOperation_t, cusparseOperation_t, Cint,
                  Cint, Cint, Ptr{cuDoubleComplex}, cusparseMatDescr_t,
                  CuPtr{cuDoubleComplex}, CuPtr{Cint}, CuPtr{Cint}, CuPtr{cuDoubleComplex},
                  Cint, csrsm2Info_t, cusparseSolvePolicy_t, CuPtr{Cvoid}),
                 handle, algo, transA, transB, m, nrhs, nnz, alpha, descrA, csrSortedValA,
                 csrSortedRowPtrA, csrSortedColIndA, B, ldb, info, policy, pBuffer)
end

function cusparseXbsrsm2_zeroPivot(handle, info, position)
    @check @runtime_ccall((:cusparseXbsrsm2_zeroPivot, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, bsrsm2Info_t, Ptr{Cint}),
                 handle, info, position)
end

function cusparseSbsrsm2_bufferSize(handle, dirA, transA, transXY, mb, n, nnzb, descrA,
                                    bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd,
                                    blockSize, info, pBufferSizeInBytes)
    @check @runtime_ccall((:cusparseSbsrsm2_bufferSize, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, cusparseOperation_t,
                  cusparseOperation_t, Cint, Cint, Cint, cusparseMatDescr_t, CuPtr{Cfloat},
                  CuPtr{Cint}, CuPtr{Cint}, Cint, bsrsm2Info_t, Ptr{Cint}),
                 handle, dirA, transA, transXY, mb, n, nnzb, descrA, bsrSortedVal,
                 bsrSortedRowPtr, bsrSortedColInd, blockSize, info, pBufferSizeInBytes)
end

function cusparseDbsrsm2_bufferSize(handle, dirA, transA, transXY, mb, n, nnzb, descrA,
                                    bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd,
                                    blockSize, info, pBufferSizeInBytes)
    @check @runtime_ccall((:cusparseDbsrsm2_bufferSize, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, cusparseOperation_t,
                  cusparseOperation_t, Cint, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{Cdouble}, CuPtr{Cint}, CuPtr{Cint}, Cint, bsrsm2Info_t, Ptr{Cint}),
                 handle, dirA, transA, transXY, mb, n, nnzb, descrA, bsrSortedVal,
                 bsrSortedRowPtr, bsrSortedColInd, blockSize, info, pBufferSizeInBytes)
end

function cusparseCbsrsm2_bufferSize(handle, dirA, transA, transXY, mb, n, nnzb, descrA,
                                    bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd,
                                    blockSize, info, pBufferSizeInBytes)
    @check @runtime_ccall((:cusparseCbsrsm2_bufferSize, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, cusparseOperation_t,
                  cusparseOperation_t, Cint, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{cuComplex}, CuPtr{Cint}, CuPtr{Cint}, Cint, bsrsm2Info_t, Ptr{Cint}),
                 handle, dirA, transA, transXY, mb, n, nnzb, descrA, bsrSortedVal,
                 bsrSortedRowPtr, bsrSortedColInd, blockSize, info, pBufferSizeInBytes)
end

function cusparseZbsrsm2_bufferSize(handle, dirA, transA, transXY, mb, n, nnzb, descrA,
                                    bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd,
                                    blockSize, info, pBufferSizeInBytes)
    @check @runtime_ccall((:cusparseZbsrsm2_bufferSize, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, cusparseOperation_t,
                  cusparseOperation_t, Cint, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{cuDoubleComplex}, CuPtr{Cint}, CuPtr{Cint}, Cint, bsrsm2Info_t,
                  Ptr{Cint}),
                 handle, dirA, transA, transXY, mb, n, nnzb, descrA, bsrSortedVal,
                 bsrSortedRowPtr, bsrSortedColInd, blockSize, info, pBufferSizeInBytes)
end

function cusparseSbsrsm2_bufferSizeExt(handle, dirA, transA, transB, mb, n, nnzb, descrA,
                                       bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd,
                                       blockSize, info, pBufferSize)
    @check @runtime_ccall((:cusparseSbsrsm2_bufferSizeExt, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, cusparseOperation_t,
                  cusparseOperation_t, Cint, Cint, Cint, cusparseMatDescr_t, CuPtr{Cfloat},
                  CuPtr{Cint}, CuPtr{Cint}, Cint, bsrsm2Info_t, CuPtr{Csize_t}),
                 handle, dirA, transA, transB, mb, n, nnzb, descrA, bsrSortedVal,
                 bsrSortedRowPtr, bsrSortedColInd, blockSize, info, pBufferSize)
end

function cusparseDbsrsm2_bufferSizeExt(handle, dirA, transA, transB, mb, n, nnzb, descrA,
                                       bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd,
                                       blockSize, info, pBufferSize)
    @check @runtime_ccall((:cusparseDbsrsm2_bufferSizeExt, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, cusparseOperation_t,
                  cusparseOperation_t, Cint, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{Cdouble}, CuPtr{Cint}, CuPtr{Cint}, Cint, bsrsm2Info_t,
                  Ptr{Csize_t}),
                 handle, dirA, transA, transB, mb, n, nnzb, descrA, bsrSortedVal,
                 bsrSortedRowPtr, bsrSortedColInd, blockSize, info, pBufferSize)
end

function cusparseCbsrsm2_bufferSizeExt(handle, dirA, transA, transB, mb, n, nnzb, descrA,
                                       bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd,
                                       blockSize, info, pBufferSize)
    @check @runtime_ccall((:cusparseCbsrsm2_bufferSizeExt, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, cusparseOperation_t,
                  cusparseOperation_t, Cint, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{cuComplex}, CuPtr{Cint}, CuPtr{Cint}, Cint, bsrsm2Info_t,
                  Ptr{Csize_t}),
                 handle, dirA, transA, transB, mb, n, nnzb, descrA, bsrSortedVal,
                 bsrSortedRowPtr, bsrSortedColInd, blockSize, info, pBufferSize)
end

function cusparseZbsrsm2_bufferSizeExt(handle, dirA, transA, transB, mb, n, nnzb, descrA,
                                       bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd,
                                       blockSize, info, pBufferSize)
    @check @runtime_ccall((:cusparseZbsrsm2_bufferSizeExt, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, cusparseOperation_t,
                  cusparseOperation_t, Cint, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{cuDoubleComplex}, CuPtr{Cint}, CuPtr{Cint}, Cint, bsrsm2Info_t,
                  Ptr{Csize_t}),
                 handle, dirA, transA, transB, mb, n, nnzb, descrA, bsrSortedVal,
                 bsrSortedRowPtr, bsrSortedColInd, blockSize, info, pBufferSize)
end

function cusparseSbsrsm2_analysis(handle, dirA, transA, transXY, mb, n, nnzb, descrA,
                                  bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd,
                                  blockSize, info, policy, pBuffer)
    @check @runtime_ccall((:cusparseSbsrsm2_analysis, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, cusparseOperation_t,
                  cusparseOperation_t, Cint, Cint, Cint, cusparseMatDescr_t, CuPtr{Cfloat},
                  CuPtr{Cint}, CuPtr{Cint}, Cint, bsrsm2Info_t, cusparseSolvePolicy_t,
                  CuPtr{Cvoid}),
                 handle, dirA, transA, transXY, mb, n, nnzb, descrA, bsrSortedVal,
                 bsrSortedRowPtr, bsrSortedColInd, blockSize, info, policy, pBuffer)
end

function cusparseDbsrsm2_analysis(handle, dirA, transA, transXY, mb, n, nnzb, descrA,
                                  bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd,
                                  blockSize, info, policy, pBuffer)
    @check @runtime_ccall((:cusparseDbsrsm2_analysis, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, cusparseOperation_t,
                  cusparseOperation_t, Cint, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{Cdouble}, CuPtr{Cint}, CuPtr{Cint}, Cint, bsrsm2Info_t,
                  cusparseSolvePolicy_t, CuPtr{Cvoid}),
                 handle, dirA, transA, transXY, mb, n, nnzb, descrA, bsrSortedVal,
                 bsrSortedRowPtr, bsrSortedColInd, blockSize, info, policy, pBuffer)
end

function cusparseCbsrsm2_analysis(handle, dirA, transA, transXY, mb, n, nnzb, descrA,
                                  bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd,
                                  blockSize, info, policy, pBuffer)
    @check @runtime_ccall((:cusparseCbsrsm2_analysis, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, cusparseOperation_t,
                  cusparseOperation_t, Cint, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{cuComplex}, CuPtr{Cint}, CuPtr{Cint}, Cint, bsrsm2Info_t,
                  cusparseSolvePolicy_t, CuPtr{Cvoid}),
                 handle, dirA, transA, transXY, mb, n, nnzb, descrA, bsrSortedVal,
                 bsrSortedRowPtr, bsrSortedColInd, blockSize, info, policy, pBuffer)
end

function cusparseZbsrsm2_analysis(handle, dirA, transA, transXY, mb, n, nnzb, descrA,
                                  bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd,
                                  blockSize, info, policy, pBuffer)
    @check @runtime_ccall((:cusparseZbsrsm2_analysis, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, cusparseOperation_t,
                  cusparseOperation_t, Cint, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{cuDoubleComplex}, CuPtr{Cint}, CuPtr{Cint}, Cint, bsrsm2Info_t,
                  cusparseSolvePolicy_t, CuPtr{Cvoid}),
                 handle, dirA, transA, transXY, mb, n, nnzb, descrA, bsrSortedVal,
                 bsrSortedRowPtr, bsrSortedColInd, blockSize, info, policy, pBuffer)
end

function cusparseSbsrsm2_solve(handle, dirA, transA, transXY, mb, n, nnzb, alpha, descrA,
                               bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockSize,
                               info, B, ldb, X, ldx, policy, pBuffer)
    @check @runtime_ccall((:cusparseSbsrsm2_solve, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, cusparseOperation_t,
                  cusparseOperation_t, Cint, Cint, Cint, Ptr{Cfloat}, cusparseMatDescr_t,
                  CuPtr{Cfloat}, CuPtr{Cint}, CuPtr{Cint}, Cint, bsrsm2Info_t,
                  CuPtr{Cfloat}, Cint, CuPtr{Cfloat}, Cint, cusparseSolvePolicy_t,
                  CuPtr{Cvoid}),
                 handle, dirA, transA, transXY, mb, n, nnzb, alpha, descrA, bsrSortedVal,
                 bsrSortedRowPtr, bsrSortedColInd, blockSize, info, B, ldb, X, ldx, policy,
                 pBuffer)
end

function cusparseDbsrsm2_solve(handle, dirA, transA, transXY, mb, n, nnzb, alpha, descrA,
                               bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockSize,
                               info, B, ldb, X, ldx, policy, pBuffer)
    @check @runtime_ccall((:cusparseDbsrsm2_solve, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, cusparseOperation_t,
                  cusparseOperation_t, Cint, Cint, Cint, Ptr{Cdouble}, cusparseMatDescr_t,
                  CuPtr{Cdouble}, CuPtr{Cint}, CuPtr{Cint}, Cint, bsrsm2Info_t,
                  CuPtr{Cdouble}, Cint, CuPtr{Cdouble}, Cint, cusparseSolvePolicy_t,
                  CuPtr{Cvoid}),
                 handle, dirA, transA, transXY, mb, n, nnzb, alpha, descrA, bsrSortedVal,
                 bsrSortedRowPtr, bsrSortedColInd, blockSize, info, B, ldb, X, ldx, policy,
                 pBuffer)
end

function cusparseCbsrsm2_solve(handle, dirA, transA, transXY, mb, n, nnzb, alpha, descrA,
                               bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockSize,
                               info, B, ldb, X, ldx, policy, pBuffer)
    @check @runtime_ccall((:cusparseCbsrsm2_solve, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, cusparseOperation_t,
                  cusparseOperation_t, Cint, Cint, Cint, Ptr{cuComplex},
                  cusparseMatDescr_t, CuPtr{cuComplex}, CuPtr{Cint}, CuPtr{Cint}, Cint,
                  bsrsm2Info_t, CuPtr{cuComplex}, Cint, CuPtr{cuComplex}, Cint,
                  cusparseSolvePolicy_t, CuPtr{Cvoid}),
                 handle, dirA, transA, transXY, mb, n, nnzb, alpha, descrA, bsrSortedVal,
                 bsrSortedRowPtr, bsrSortedColInd, blockSize, info, B, ldb, X, ldx, policy,
                 pBuffer)
end

function cusparseZbsrsm2_solve(handle, dirA, transA, transXY, mb, n, nnzb, alpha, descrA,
                               bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd, blockSize,
                               info, B, ldb, X, ldx, policy, pBuffer)
    @check @runtime_ccall((:cusparseZbsrsm2_solve, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, cusparseOperation_t,
                  cusparseOperation_t, Cint, Cint, Cint, Ptr{cuDoubleComplex},
                  cusparseMatDescr_t, CuPtr{cuDoubleComplex}, CuPtr{Cint}, CuPtr{Cint},
                  Cint, bsrsm2Info_t, CuPtr{cuDoubleComplex}, Cint, CuPtr{cuDoubleComplex},
                  Cint, cusparseSolvePolicy_t, CuPtr{Cvoid}),
                 handle, dirA, transA, transXY, mb, n, nnzb, alpha, descrA, bsrSortedVal,
                 bsrSortedRowPtr, bsrSortedColInd, blockSize, info, B, ldb, X, ldx, policy,
                 pBuffer)
end

function cusparseCsrilu0Ex(handle, trans, m, descrA, csrSortedValA_ValM,
                           csrSortedValA_ValMtype, csrSortedRowPtrA, csrSortedColIndA,
                           info, executiontype)
    @check @runtime_ccall((:cusparseCsrilu0Ex, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseOperation_t, Cint, cusparseMatDescr_t,
                  CuPtr{Cvoid}, cudaDataType, CuPtr{Cint}, CuPtr{Cint},
                  cusparseSolveAnalysisInfo_t, cudaDataType),
                 handle, trans, m, descrA, csrSortedValA_ValM, csrSortedValA_ValMtype,
                 csrSortedRowPtrA, csrSortedColIndA, info, executiontype)
end

function cusparseScsrilu0(handle, trans, m, descrA, csrSortedValA_ValM, csrSortedRowPtrA,
                          csrSortedColIndA, info)
    @check @runtime_ccall((:cusparseScsrilu0, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseOperation_t, Cint, cusparseMatDescr_t,
                  CuPtr{Cfloat}, CuPtr{Cint}, CuPtr{Cint}, cusparseSolveAnalysisInfo_t),
                 handle, trans, m, descrA, csrSortedValA_ValM, csrSortedRowPtrA,
                 csrSortedColIndA, info)
end

function cusparseDcsrilu0(handle, trans, m, descrA, csrSortedValA_ValM, csrSortedRowPtrA,
                          csrSortedColIndA, info)
    @check @runtime_ccall((:cusparseDcsrilu0, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseOperation_t, Cint, cusparseMatDescr_t,
                  CuPtr{Cdouble}, CuPtr{Cint}, CuPtr{Cint}, cusparseSolveAnalysisInfo_t),
                 handle, trans, m, descrA, csrSortedValA_ValM, csrSortedRowPtrA,
                 csrSortedColIndA, info)
end

function cusparseCcsrilu0(handle, trans, m, descrA, csrSortedValA_ValM, csrSortedRowPtrA,
                          csrSortedColIndA, info)
    @check @runtime_ccall((:cusparseCcsrilu0, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseOperation_t, Cint, cusparseMatDescr_t,
                  CuPtr{cuComplex}, CuPtr{Cint}, CuPtr{Cint}, cusparseSolveAnalysisInfo_t),
                 handle, trans, m, descrA, csrSortedValA_ValM, csrSortedRowPtrA,
                 csrSortedColIndA, info)
end

function cusparseZcsrilu0(handle, trans, m, descrA, csrSortedValA_ValM, csrSortedRowPtrA,
                          csrSortedColIndA, info)
    @check @runtime_ccall((:cusparseZcsrilu0, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseOperation_t, Cint, cusparseMatDescr_t,
                  CuPtr{cuDoubleComplex}, CuPtr{Cint}, CuPtr{Cint},
                  cusparseSolveAnalysisInfo_t),
                 handle, trans, m, descrA, csrSortedValA_ValM, csrSortedRowPtrA,
                 csrSortedColIndA, info)
end

function cusparseScsrilu02_numericBoost(handle, info, enable_boost, tol, boost_val)
    @check @runtime_ccall((:cusparseScsrilu02_numericBoost, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, csrilu02Info_t, Cint, Ptr{Cdouble}, Ptr{Cfloat}),
                 handle, info, enable_boost, tol, boost_val)
end

function cusparseDcsrilu02_numericBoost(handle, info, enable_boost, tol, boost_val)
    @check @runtime_ccall((:cusparseDcsrilu02_numericBoost, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, csrilu02Info_t, Cint, Ptr{Cdouble}, Ptr{Cdouble}),
                 handle, info, enable_boost, tol, boost_val)
end

function cusparseCcsrilu02_numericBoost(handle, info, enable_boost, tol, boost_val)
    @check @runtime_ccall((:cusparseCcsrilu02_numericBoost, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, csrilu02Info_t, Cint, Ptr{Cdouble}, Ptr{cuComplex}),
                 handle, info, enable_boost, tol, boost_val)
end

function cusparseZcsrilu02_numericBoost(handle, info, enable_boost, tol, boost_val)
    @check @runtime_ccall((:cusparseZcsrilu02_numericBoost, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, csrilu02Info_t, Cint, Ptr{Cdouble},
                  Ptr{cuDoubleComplex}),
                 handle, info, enable_boost, tol, boost_val)
end

function cusparseXcsrilu02_zeroPivot(handle, info, position)
    @check @runtime_ccall((:cusparseXcsrilu02_zeroPivot, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, csrilu02Info_t, Ptr{Cint}),
                 handle, info, position)
end

function cusparseScsrilu02_bufferSize(handle, m, nnz, descrA, csrSortedValA,
                                      csrSortedRowPtrA, csrSortedColIndA, info,
                                      pBufferSizeInBytes)
    @check @runtime_ccall((:cusparseScsrilu02_bufferSize, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, cusparseMatDescr_t, CuPtr{Cfloat},
                  CuPtr{Cint}, CuPtr{Cint}, csrilu02Info_t, Ptr{Cint}),
                 handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA,
                 info, pBufferSizeInBytes)
end

function cusparseDcsrilu02_bufferSize(handle, m, nnz, descrA, csrSortedValA,
                                      csrSortedRowPtrA, csrSortedColIndA, info,
                                      pBufferSizeInBytes)
    @check @runtime_ccall((:cusparseDcsrilu02_bufferSize, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, cusparseMatDescr_t, CuPtr{Cdouble},
                  CuPtr{Cint}, CuPtr{Cint}, csrilu02Info_t, Ptr{Cint}),
                 handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA,
                 info, pBufferSizeInBytes)
end

function cusparseCcsrilu02_bufferSize(handle, m, nnz, descrA, csrSortedValA,
                                      csrSortedRowPtrA, csrSortedColIndA, info,
                                      pBufferSizeInBytes)
    @check @runtime_ccall((:cusparseCcsrilu02_bufferSize, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, cusparseMatDescr_t, CuPtr{cuComplex},
                  CuPtr{Cint}, CuPtr{Cint}, csrilu02Info_t, Ptr{Cint}),
                 handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA,
                 info, pBufferSizeInBytes)
end

function cusparseZcsrilu02_bufferSize(handle, m, nnz, descrA, csrSortedValA,
                                      csrSortedRowPtrA, csrSortedColIndA, info,
                                      pBufferSizeInBytes)
    @check @runtime_ccall((:cusparseZcsrilu02_bufferSize, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{cuDoubleComplex}, CuPtr{Cint}, CuPtr{Cint}, csrilu02Info_t,
                  Ptr{Cint}),
                 handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA,
                 info, pBufferSizeInBytes)
end

function cusparseScsrilu02_bufferSizeExt(handle, m, nnz, descrA, csrSortedVal,
                                         csrSortedRowPtr, csrSortedColInd, info, pBufferSize)
    @check @runtime_ccall((:cusparseScsrilu02_bufferSizeExt, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, cusparseMatDescr_t, CuPtr{Cfloat},
                  CuPtr{Cint}, CuPtr{Cint}, csrilu02Info_t, Ptr{Csize_t}),
                 handle, m, nnz, descrA, csrSortedVal, csrSortedRowPtr, csrSortedColInd,
                 info, pBufferSize)
end

function cusparseDcsrilu02_bufferSizeExt(handle, m, nnz, descrA, csrSortedVal,
                                         csrSortedRowPtr, csrSortedColInd, info, pBufferSize)
    @check @runtime_ccall((:cusparseDcsrilu02_bufferSizeExt, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, cusparseMatDescr_t, CuPtr{Cdouble},
                  CuPtr{Cint}, CuPtr{Cint}, csrilu02Info_t, Ptr{Csize_t}),
                 handle, m, nnz, descrA, csrSortedVal, csrSortedRowPtr, csrSortedColInd,
                 info, pBufferSize)
end

function cusparseCcsrilu02_bufferSizeExt(handle, m, nnz, descrA, csrSortedVal,
                                         csrSortedRowPtr, csrSortedColInd, info, pBufferSize)
    @check @runtime_ccall((:cusparseCcsrilu02_bufferSizeExt, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, cusparseMatDescr_t, CuPtr{cuComplex},
                  CuPtr{Cint}, CuPtr{Cint}, csrilu02Info_t, Ptr{Csize_t}),
                 handle, m, nnz, descrA, csrSortedVal, csrSortedRowPtr, csrSortedColInd,
                 info, pBufferSize)
end

function cusparseZcsrilu02_bufferSizeExt(handle, m, nnz, descrA, csrSortedVal,
                                         csrSortedRowPtr, csrSortedColInd, info, pBufferSize)
    @check @runtime_ccall((:cusparseZcsrilu02_bufferSizeExt, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{cuDoubleComplex}, CuPtr{Cint}, CuPtr{Cint}, csrilu02Info_t,
                  Ptr{Csize_t}),
                 handle, m, nnz, descrA, csrSortedVal, csrSortedRowPtr, csrSortedColInd,
                 info, pBufferSize)
end

function cusparseScsrilu02_analysis(handle, m, nnz, descrA, csrSortedValA,
                                    csrSortedRowPtrA, csrSortedColIndA, info, policy,
                                    pBuffer)
    @check @runtime_ccall((:cusparseScsrilu02_analysis, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, cusparseMatDescr_t, CuPtr{Cfloat},
                  CuPtr{Cint}, CuPtr{Cint}, csrilu02Info_t, cusparseSolvePolicy_t,
                  CuPtr{Cvoid}),
                 handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA,
                 info, policy, pBuffer)
end

function cusparseDcsrilu02_analysis(handle, m, nnz, descrA, csrSortedValA,
                                    csrSortedRowPtrA, csrSortedColIndA, info, policy,
                                    pBuffer)
    @check @runtime_ccall((:cusparseDcsrilu02_analysis, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, cusparseMatDescr_t, CuPtr{Cdouble},
                  CuPtr{Cint}, CuPtr{Cint}, csrilu02Info_t, cusparseSolvePolicy_t,
                  CuPtr{Cvoid}),
                 handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA,
                 info, policy, pBuffer)
end

function cusparseCcsrilu02_analysis(handle, m, nnz, descrA, csrSortedValA,
                                    csrSortedRowPtrA, csrSortedColIndA, info, policy,
                                    pBuffer)
    @check @runtime_ccall((:cusparseCcsrilu02_analysis, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, cusparseMatDescr_t, CuPtr{cuComplex},
                  CuPtr{Cint}, CuPtr{Cint}, csrilu02Info_t, cusparseSolvePolicy_t,
                  CuPtr{Cvoid}),
                 handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA,
                 info, policy, pBuffer)
end

function cusparseZcsrilu02_analysis(handle, m, nnz, descrA, csrSortedValA,
                                    csrSortedRowPtrA, csrSortedColIndA, info, policy,
                                    pBuffer)
    @check @runtime_ccall((:cusparseZcsrilu02_analysis, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{cuDoubleComplex}, CuPtr{Cint}, CuPtr{Cint}, csrilu02Info_t,
                  cusparseSolvePolicy_t, CuPtr{Cvoid}),
                 handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA,
                 info, policy, pBuffer)
end

function cusparseScsrilu02(handle, m, nnz, descrA, csrSortedValA_valM, csrSortedRowPtrA,
                           csrSortedColIndA, info, policy, pBuffer)
    @check @runtime_ccall((:cusparseScsrilu02, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, cusparseMatDescr_t, CuPtr{Cfloat},
                  CuPtr{Cint}, CuPtr{Cint}, csrilu02Info_t, cusparseSolvePolicy_t,
                  CuPtr{Cvoid}),
                 handle, m, nnz, descrA, csrSortedValA_valM, csrSortedRowPtrA,
                 csrSortedColIndA, info, policy, pBuffer)
end

function cusparseDcsrilu02(handle, m, nnz, descrA, csrSortedValA_valM, csrSortedRowPtrA,
                           csrSortedColIndA, info, policy, pBuffer)
    @check @runtime_ccall((:cusparseDcsrilu02, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, cusparseMatDescr_t, CuPtr{Cdouble},
                  CuPtr{Cint}, CuPtr{Cint}, csrilu02Info_t, cusparseSolvePolicy_t,
                  CuPtr{Cvoid}),
                 handle, m, nnz, descrA, csrSortedValA_valM, csrSortedRowPtrA,
                 csrSortedColIndA, info, policy, pBuffer)
end

function cusparseCcsrilu02(handle, m, nnz, descrA, csrSortedValA_valM, csrSortedRowPtrA,
                           csrSortedColIndA, info, policy, pBuffer)
    @check @runtime_ccall((:cusparseCcsrilu02, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, cusparseMatDescr_t, CuPtr{cuComplex},
                  CuPtr{Cint}, CuPtr{Cint}, csrilu02Info_t, cusparseSolvePolicy_t,
                  CuPtr{Cvoid}),
                 handle, m, nnz, descrA, csrSortedValA_valM, csrSortedRowPtrA,
                 csrSortedColIndA, info, policy, pBuffer)
end

function cusparseZcsrilu02(handle, m, nnz, descrA, csrSortedValA_valM, csrSortedRowPtrA,
                           csrSortedColIndA, info, policy, pBuffer)
    @check @runtime_ccall((:cusparseZcsrilu02, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{cuDoubleComplex}, CuPtr{Cint}, CuPtr{Cint}, csrilu02Info_t,
                  cusparseSolvePolicy_t, CuPtr{Cvoid}),
                 handle, m, nnz, descrA, csrSortedValA_valM, csrSortedRowPtrA,
                 csrSortedColIndA, info, policy, pBuffer)
end

function cusparseSbsrilu02_numericBoost(handle, info, enable_boost, tol, boost_val)
    @check @runtime_ccall((:cusparseSbsrilu02_numericBoost, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, bsrilu02Info_t, Cint, Ptr{Cdouble}, Ptr{Cfloat}),
                 handle, info, enable_boost, tol, boost_val)
end

function cusparseDbsrilu02_numericBoost(handle, info, enable_boost, tol, boost_val)
    @check @runtime_ccall((:cusparseDbsrilu02_numericBoost, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, bsrilu02Info_t, Cint, Ptr{Cdouble}, Ptr{Cdouble}),
                 handle, info, enable_boost, tol, boost_val)
end

function cusparseCbsrilu02_numericBoost(handle, info, enable_boost, tol, boost_val)
    @check @runtime_ccall((:cusparseCbsrilu02_numericBoost, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, bsrilu02Info_t, Cint, Ptr{Cdouble}, Ptr{cuComplex}),
                 handle, info, enable_boost, tol, boost_val)
end

function cusparseZbsrilu02_numericBoost(handle, info, enable_boost, tol, boost_val)
    @check @runtime_ccall((:cusparseZbsrilu02_numericBoost, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, bsrilu02Info_t, Cint, Ptr{Cdouble},
                  Ptr{cuDoubleComplex}),
                 handle, info, enable_boost, tol, boost_val)
end

function cusparseXbsrilu02_zeroPivot(handle, info, position)
    @check @runtime_ccall((:cusparseXbsrilu02_zeroPivot, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, bsrilu02Info_t, Ptr{Cint}),
                 handle, info, position)
end

function cusparseSbsrilu02_bufferSize(handle, dirA, mb, nnzb, descrA, bsrSortedVal,
                                      bsrSortedRowPtr, bsrSortedColInd, blockDim, info,
                                      pBufferSizeInBytes)
    @check @runtime_ccall((:cusparseSbsrilu02_bufferSize, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{Cfloat}, CuPtr{Cint}, CuPtr{Cint}, Cint, bsrilu02Info_t, Ptr{Cint}),
                 handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr,
                 bsrSortedColInd, blockDim, info, pBufferSizeInBytes)
end

function cusparseDbsrilu02_bufferSize(handle, dirA, mb, nnzb, descrA, bsrSortedVal,
                                      bsrSortedRowPtr, bsrSortedColInd, blockDim, info,
                                      pBufferSizeInBytes)
    @check @runtime_ccall((:cusparseDbsrilu02_bufferSize, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{Cdouble}, CuPtr{Cint}, CuPtr{Cint}, Cint, bsrilu02Info_t, Ptr{Cint}),
                 handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr,
                 bsrSortedColInd, blockDim, info, pBufferSizeInBytes)
end

function cusparseCbsrilu02_bufferSize(handle, dirA, mb, nnzb, descrA, bsrSortedVal,
                                      bsrSortedRowPtr, bsrSortedColInd, blockDim, info,
                                      pBufferSizeInBytes)
    @check @runtime_ccall((:cusparseCbsrilu02_bufferSize, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{cuComplex}, CuPtr{Cint}, CuPtr{Cint}, Cint, bsrilu02Info_t,
                  Ptr{Cint}),
                 handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr,
                 bsrSortedColInd, blockDim, info, pBufferSizeInBytes)
end

function cusparseZbsrilu02_bufferSize(handle, dirA, mb, nnzb, descrA, bsrSortedVal,
                                      bsrSortedRowPtr, bsrSortedColInd, blockDim, info,
                                      pBufferSizeInBytes)
    @check @runtime_ccall((:cusparseZbsrilu02_bufferSize, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{cuDoubleComplex}, CuPtr{Cint}, CuPtr{Cint}, Cint, bsrilu02Info_t,
                  Ptr{Cint}),
                 handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr,
                 bsrSortedColInd, blockDim, info, pBufferSizeInBytes)
end

function cusparseSbsrilu02_bufferSizeExt(handle, dirA, mb, nnzb, descrA, bsrSortedVal,
                                         bsrSortedRowPtr, bsrSortedColInd, blockSize, info,
                                         pBufferSize)
    @check @runtime_ccall((:cusparseSbsrilu02_bufferSizeExt, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{Cfloat}, CuPtr{Cint}, CuPtr{Cint}, Cint, bsrilu02Info_t,
                  Ptr{Csize_t}),
                 handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr,
                 bsrSortedColInd, blockSize, info, pBufferSize)
end

function cusparseDbsrilu02_bufferSizeExt(handle, dirA, mb, nnzb, descrA, bsrSortedVal,
                                         bsrSortedRowPtr, bsrSortedColInd, blockSize, info,
                                         pBufferSize)
    @check @runtime_ccall((:cusparseDbsrilu02_bufferSizeExt, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{Cdouble}, CuPtr{Cint}, CuPtr{Cint}, Cint, bsrilu02Info_t,
                  Ptr{Csize_t}),
                 handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr,
                 bsrSortedColInd, blockSize, info, pBufferSize)
end

function cusparseCbsrilu02_bufferSizeExt(handle, dirA, mb, nnzb, descrA, bsrSortedVal,
                                         bsrSortedRowPtr, bsrSortedColInd, blockSize, info,
                                         pBufferSize)
    @check @runtime_ccall((:cusparseCbsrilu02_bufferSizeExt, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{cuComplex}, CuPtr{Cint}, CuPtr{Cint}, Cint, bsrilu02Info_t,
                  Ptr{Csize_t}),
                 handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr,
                 bsrSortedColInd, blockSize, info, pBufferSize)
end

function cusparseZbsrilu02_bufferSizeExt(handle, dirA, mb, nnzb, descrA, bsrSortedVal,
                                         bsrSortedRowPtr, bsrSortedColInd, blockSize, info,
                                         pBufferSize)
    @check @runtime_ccall((:cusparseZbsrilu02_bufferSizeExt, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{cuDoubleComplex}, CuPtr{Cint}, CuPtr{Cint}, Cint, bsrilu02Info_t,
                  Ptr{Csize_t}),
                 handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr,
                 bsrSortedColInd, blockSize, info, pBufferSize)
end

function cusparseSbsrilu02_analysis(handle, dirA, mb, nnzb, descrA, bsrSortedVal,
                                    bsrSortedRowPtr, bsrSortedColInd, blockDim, info,
                                    policy, pBuffer)
    @check @runtime_ccall((:cusparseSbsrilu02_analysis, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{Cfloat}, CuPtr{Cint}, CuPtr{Cint}, Cint, bsrilu02Info_t,
                  cusparseSolvePolicy_t, CuPtr{Cvoid}),
                 handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr,
                 bsrSortedColInd, blockDim, info, policy, pBuffer)
end

function cusparseDbsrilu02_analysis(handle, dirA, mb, nnzb, descrA, bsrSortedVal,
                                    bsrSortedRowPtr, bsrSortedColInd, blockDim, info,
                                    policy, pBuffer)
    @check @runtime_ccall((:cusparseDbsrilu02_analysis, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{Cdouble}, CuPtr{Cint}, CuPtr{Cint}, Cint, bsrilu02Info_t,
                  cusparseSolvePolicy_t, CuPtr{Cvoid}),
                 handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr,
                 bsrSortedColInd, blockDim, info, policy, pBuffer)
end

function cusparseCbsrilu02_analysis(handle, dirA, mb, nnzb, descrA, bsrSortedVal,
                                    bsrSortedRowPtr, bsrSortedColInd, blockDim, info,
                                    policy, pBuffer)
    @check @runtime_ccall((:cusparseCbsrilu02_analysis, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{cuComplex}, CuPtr{Cint}, CuPtr{Cint}, Cint, bsrilu02Info_t,
                  cusparseSolvePolicy_t, CuPtr{Cvoid}),
                 handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr,
                 bsrSortedColInd, blockDim, info, policy, pBuffer)
end

function cusparseZbsrilu02_analysis(handle, dirA, mb, nnzb, descrA, bsrSortedVal,
                                    bsrSortedRowPtr, bsrSortedColInd, blockDim, info,
                                    policy, pBuffer)
    @check @runtime_ccall((:cusparseZbsrilu02_analysis, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{cuDoubleComplex}, CuPtr{Cint}, CuPtr{Cint}, Cint, bsrilu02Info_t,
                  cusparseSolvePolicy_t, CuPtr{Cvoid}),
                 handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr,
                 bsrSortedColInd, blockDim, info, policy, pBuffer)
end

function cusparseSbsrilu02(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr,
                           bsrSortedColInd, blockDim, info, policy, pBuffer)
    @check @runtime_ccall((:cusparseSbsrilu02, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{Cfloat}, CuPtr{Cint}, CuPtr{Cint}, Cint, bsrilu02Info_t,
                  cusparseSolvePolicy_t, CuPtr{Cvoid}),
                 handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr,
                 bsrSortedColInd, blockDim, info, policy, pBuffer)
end

function cusparseDbsrilu02(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr,
                           bsrSortedColInd, blockDim, info, policy, pBuffer)
    @check @runtime_ccall((:cusparseDbsrilu02, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{Cdouble}, CuPtr{Cint}, CuPtr{Cint}, Cint, bsrilu02Info_t,
                  cusparseSolvePolicy_t, CuPtr{Cvoid}),
                 handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr,
                 bsrSortedColInd, blockDim, info, policy, pBuffer)
end

function cusparseCbsrilu02(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr,
                           bsrSortedColInd, blockDim, info, policy, pBuffer)
    @check @runtime_ccall((:cusparseCbsrilu02, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{cuComplex}, CuPtr{Cint}, CuPtr{Cint}, Cint, bsrilu02Info_t,
                  cusparseSolvePolicy_t, CuPtr{Cvoid}),
                 handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr,
                 bsrSortedColInd, blockDim, info, policy, pBuffer)
end

function cusparseZbsrilu02(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr,
                           bsrSortedColInd, blockDim, info, policy, pBuffer)
    @check @runtime_ccall((:cusparseZbsrilu02, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{cuDoubleComplex}, CuPtr{Cint}, CuPtr{Cint}, Cint, bsrilu02Info_t,
                  cusparseSolvePolicy_t, CuPtr{Cvoid}),
                 handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr,
                 bsrSortedColInd, blockDim, info, policy, pBuffer)
end

function cusparseScsric0(handle, trans, m, descrA, csrSortedValA_ValM, csrSortedRowPtrA,
                         csrSortedColIndA, info)
    @check @runtime_ccall((:cusparseScsric0, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseOperation_t, Cint, cusparseMatDescr_t,
                  CuPtr{Cfloat}, CuPtr{Cint}, CuPtr{Cint}, cusparseSolveAnalysisInfo_t),
                 handle, trans, m, descrA, csrSortedValA_ValM, csrSortedRowPtrA,
                 csrSortedColIndA, info)
end

function cusparseDcsric0(handle, trans, m, descrA, csrSortedValA_ValM, csrSortedRowPtrA,
                         csrSortedColIndA, info)
    @check @runtime_ccall((:cusparseDcsric0, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseOperation_t, Cint, cusparseMatDescr_t,
                  CuPtr{Cdouble}, CuPtr{Cint}, CuPtr{Cint}, cusparseSolveAnalysisInfo_t),
                 handle, trans, m, descrA, csrSortedValA_ValM, csrSortedRowPtrA,
                 csrSortedColIndA, info)
end

function cusparseCcsric0(handle, trans, m, descrA, csrSortedValA_ValM, csrSortedRowPtrA,
                         csrSortedColIndA, info)
    @check @runtime_ccall((:cusparseCcsric0, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseOperation_t, Cint, cusparseMatDescr_t,
                  CuPtr{cuComplex}, CuPtr{Cint}, CuPtr{Cint}, cusparseSolveAnalysisInfo_t),
                 handle, trans, m, descrA, csrSortedValA_ValM, csrSortedRowPtrA,
                 csrSortedColIndA, info)
end

function cusparseZcsric0(handle, trans, m, descrA, csrSortedValA_ValM, csrSortedRowPtrA,
                         csrSortedColIndA, info)
    @check @runtime_ccall((:cusparseZcsric0, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseOperation_t, Cint, cusparseMatDescr_t,
                  CuPtr{cuDoubleComplex}, CuPtr{Cint}, CuPtr{Cint},
                  cusparseSolveAnalysisInfo_t),
                 handle, trans, m, descrA, csrSortedValA_ValM, csrSortedRowPtrA,
                 csrSortedColIndA, info)
end

function cusparseXcsric02_zeroPivot(handle, info, position)
    @check @runtime_ccall((:cusparseXcsric02_zeroPivot, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, csric02Info_t, Ptr{Cint}),
                 handle, info, position)
end

function cusparseScsric02_bufferSize(handle, m, nnz, descrA, csrSortedValA,
                                     csrSortedRowPtrA, csrSortedColIndA, info,
                                     pBufferSizeInBytes)
    @check @runtime_ccall((:cusparseScsric02_bufferSize, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, cusparseMatDescr_t, CuPtr{Cfloat},
                  CuPtr{Cint}, CuPtr{Cint}, csric02Info_t, Ptr{Cint}),
                 handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA,
                 info, pBufferSizeInBytes)
end

function cusparseDcsric02_bufferSize(handle, m, nnz, descrA, csrSortedValA,
                                     csrSortedRowPtrA, csrSortedColIndA, info,
                                     pBufferSizeInBytes)
    @check @runtime_ccall((:cusparseDcsric02_bufferSize, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, cusparseMatDescr_t, CuPtr{Cdouble},
                  CuPtr{Cint}, CuPtr{Cint}, csric02Info_t, Ptr{Cint}),
                 handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA,
                 info, pBufferSizeInBytes)
end

function cusparseCcsric02_bufferSize(handle, m, nnz, descrA, csrSortedValA,
                                     csrSortedRowPtrA, csrSortedColIndA, info,
                                     pBufferSizeInBytes)
    @check @runtime_ccall((:cusparseCcsric02_bufferSize, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, cusparseMatDescr_t, CuPtr{cuComplex},
                  CuPtr{Cint}, CuPtr{Cint}, csric02Info_t, Ptr{Cint}),
                 handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA,
                 info, pBufferSizeInBytes)
end

function cusparseZcsric02_bufferSize(handle, m, nnz, descrA, csrSortedValA,
                                     csrSortedRowPtrA, csrSortedColIndA, info,
                                     pBufferSizeInBytes)
    @check @runtime_ccall((:cusparseZcsric02_bufferSize, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{cuDoubleComplex}, CuPtr{Cint}, CuPtr{Cint}, csric02Info_t,
                  Ptr{Cint}),
                 handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA,
                 info, pBufferSizeInBytes)
end

function cusparseScsric02_bufferSizeExt(handle, m, nnz, descrA, csrSortedVal,
                                        csrSortedRowPtr, csrSortedColInd, info, pBufferSize)
    @check @runtime_ccall((:cusparseScsric02_bufferSizeExt, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, cusparseMatDescr_t, CuPtr{Cfloat},
                  CuPtr{Cint}, CuPtr{Cint}, csric02Info_t, Ptr{Csize_t}),
                 handle, m, nnz, descrA, csrSortedVal, csrSortedRowPtr, csrSortedColInd,
                 info, pBufferSize)
end

function cusparseDcsric02_bufferSizeExt(handle, m, nnz, descrA, csrSortedVal,
                                        csrSortedRowPtr, csrSortedColInd, info, pBufferSize)
    @check @runtime_ccall((:cusparseDcsric02_bufferSizeExt, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, cusparseMatDescr_t, CuPtr{Cdouble},
                  CuPtr{Cint}, CuPtr{Cint}, csric02Info_t, Ptr{Csize_t}),
                 handle, m, nnz, descrA, csrSortedVal, csrSortedRowPtr, csrSortedColInd,
                 info, pBufferSize)
end

function cusparseCcsric02_bufferSizeExt(handle, m, nnz, descrA, csrSortedVal,
                                        csrSortedRowPtr, csrSortedColInd, info, pBufferSize)
    @check @runtime_ccall((:cusparseCcsric02_bufferSizeExt, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, cusparseMatDescr_t, CuPtr{cuComplex},
                  CuPtr{Cint}, CuPtr{Cint}, csric02Info_t, Ptr{Csize_t}),
                 handle, m, nnz, descrA, csrSortedVal, csrSortedRowPtr, csrSortedColInd,
                 info, pBufferSize)
end

function cusparseZcsric02_bufferSizeExt(handle, m, nnz, descrA, csrSortedVal,
                                        csrSortedRowPtr, csrSortedColInd, info, pBufferSize)
    @check @runtime_ccall((:cusparseZcsric02_bufferSizeExt, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{cuDoubleComplex}, CuPtr{Cint}, CuPtr{Cint}, csric02Info_t,
                  Ptr{Csize_t}),
                 handle, m, nnz, descrA, csrSortedVal, csrSortedRowPtr, csrSortedColInd,
                 info, pBufferSize)
end

function cusparseScsric02_analysis(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA,
                                   csrSortedColIndA, info, policy, pBuffer)
    @check @runtime_ccall((:cusparseScsric02_analysis, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, cusparseMatDescr_t, CuPtr{Cfloat},
                  CuPtr{Cint}, CuPtr{Cint}, csric02Info_t, cusparseSolvePolicy_t,
                  CuPtr{Cvoid}),
                 handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA,
                 info, policy, pBuffer)
end

function cusparseDcsric02_analysis(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA,
                                   csrSortedColIndA, info, policy, pBuffer)
    @check @runtime_ccall((:cusparseDcsric02_analysis, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, cusparseMatDescr_t, CuPtr{Cdouble},
                  CuPtr{Cint}, CuPtr{Cint}, csric02Info_t, cusparseSolvePolicy_t,
                  CuPtr{Cvoid}),
                 handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA,
                 info, policy, pBuffer)
end

function cusparseCcsric02_analysis(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA,
                                   csrSortedColIndA, info, policy, pBuffer)
    @check @runtime_ccall((:cusparseCcsric02_analysis, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, cusparseMatDescr_t, CuPtr{cuComplex},
                  CuPtr{Cint}, CuPtr{Cint}, csric02Info_t, cusparseSolvePolicy_t,
                  CuPtr{Cvoid}),
                 handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA,
                 info, policy, pBuffer)
end

function cusparseZcsric02_analysis(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA,
                                   csrSortedColIndA, info, policy, pBuffer)
    @check @runtime_ccall((:cusparseZcsric02_analysis, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{cuDoubleComplex}, CuPtr{Cint}, CuPtr{Cint}, csric02Info_t,
                  cusparseSolvePolicy_t, CuPtr{Cvoid}),
                 handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA,
                 info, policy, pBuffer)
end

function cusparseScsric02(handle, m, nnz, descrA, csrSortedValA_valM, csrSortedRowPtrA,
                          csrSortedColIndA, info, policy, pBuffer)
    @check @runtime_ccall((:cusparseScsric02, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, cusparseMatDescr_t, CuPtr{Cfloat},
                  CuPtr{Cint}, CuPtr{Cint}, csric02Info_t, cusparseSolvePolicy_t,
                  CuPtr{Cvoid}),
                 handle, m, nnz, descrA, csrSortedValA_valM, csrSortedRowPtrA,
                 csrSortedColIndA, info, policy, pBuffer)
end

function cusparseDcsric02(handle, m, nnz, descrA, csrSortedValA_valM, csrSortedRowPtrA,
                          csrSortedColIndA, info, policy, pBuffer)
    @check @runtime_ccall((:cusparseDcsric02, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, cusparseMatDescr_t, CuPtr{Cdouble},
                  CuPtr{Cint}, CuPtr{Cint}, csric02Info_t, cusparseSolvePolicy_t,
                  CuPtr{Cvoid}),
                 handle, m, nnz, descrA, csrSortedValA_valM, csrSortedRowPtrA,
                 csrSortedColIndA, info, policy, pBuffer)
end

function cusparseCcsric02(handle, m, nnz, descrA, csrSortedValA_valM, csrSortedRowPtrA,
                          csrSortedColIndA, info, policy, pBuffer)
    @check @runtime_ccall((:cusparseCcsric02, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, cusparseMatDescr_t, CuPtr{cuComplex},
                  CuPtr{Cint}, CuPtr{Cint}, csric02Info_t, cusparseSolvePolicy_t,
                  CuPtr{Cvoid}),
                 handle, m, nnz, descrA, csrSortedValA_valM, csrSortedRowPtrA,
                 csrSortedColIndA, info, policy, pBuffer)
end

function cusparseZcsric02(handle, m, nnz, descrA, csrSortedValA_valM, csrSortedRowPtrA,
                          csrSortedColIndA, info, policy, pBuffer)
    @check @runtime_ccall((:cusparseZcsric02, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{cuDoubleComplex}, CuPtr{Cint}, CuPtr{Cint}, csric02Info_t,
                  cusparseSolvePolicy_t, CuPtr{Cvoid}),
                 handle, m, nnz, descrA, csrSortedValA_valM, csrSortedRowPtrA,
                 csrSortedColIndA, info, policy, pBuffer)
end

function cusparseXbsric02_zeroPivot(handle, info, position)
    @check @runtime_ccall((:cusparseXbsric02_zeroPivot, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, bsric02Info_t, Ptr{Cint}),
                 handle, info, position)
end

function cusparseSbsric02_bufferSize(handle, dirA, mb, nnzb, descrA, bsrSortedVal,
                                     bsrSortedRowPtr, bsrSortedColInd, blockDim, info,
                                     pBufferSizeInBytes)
    @check @runtime_ccall((:cusparseSbsric02_bufferSize, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{Cfloat}, CuPtr{Cint}, CuPtr{Cint}, Cint, bsric02Info_t, Ptr{Cint}),
                 handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr,
                 bsrSortedColInd, blockDim, info, pBufferSizeInBytes)
end

function cusparseDbsric02_bufferSize(handle, dirA, mb, nnzb, descrA, bsrSortedVal,
                                     bsrSortedRowPtr, bsrSortedColInd, blockDim, info,
                                     pBufferSizeInBytes)
    @check @runtime_ccall((:cusparseDbsric02_bufferSize, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{Cdouble}, CuPtr{Cint}, CuPtr{Cint}, Cint, bsric02Info_t, Ptr{Cint}),
                 handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr,
                 bsrSortedColInd, blockDim, info, pBufferSizeInBytes)
end

function cusparseCbsric02_bufferSize(handle, dirA, mb, nnzb, descrA, bsrSortedVal,
                                     bsrSortedRowPtr, bsrSortedColInd, blockDim, info,
                                     pBufferSizeInBytes)
    @check @runtime_ccall((:cusparseCbsric02_bufferSize, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{cuComplex}, CuPtr{Cint}, CuPtr{Cint}, Cint, bsric02Info_t,
                  Ptr{Cint}),
                 handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr,
                 bsrSortedColInd, blockDim, info, pBufferSizeInBytes)
end

function cusparseZbsric02_bufferSize(handle, dirA, mb, nnzb, descrA, bsrSortedVal,
                                     bsrSortedRowPtr, bsrSortedColInd, blockDim, info,
                                     pBufferSizeInBytes)
    @check @runtime_ccall((:cusparseZbsric02_bufferSize, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{cuDoubleComplex}, CuPtr{Cint}, CuPtr{Cint}, Cint, bsric02Info_t,
                  Ptr{Cint}),
                 handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr,
                 bsrSortedColInd, blockDim, info, pBufferSizeInBytes)
end

function cusparseSbsric02_bufferSizeExt(handle, dirA, mb, nnzb, descrA, bsrSortedVal,
                                        bsrSortedRowPtr, bsrSortedColInd, blockSize, info,
                                        pBufferSize)
    @check @runtime_ccall((:cusparseSbsric02_bufferSizeExt, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{Cfloat}, CuPtr{Cint}, CuPtr{Cint}, Cint, bsric02Info_t,
                  Ptr{Csize_t}),
                 handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr,
                 bsrSortedColInd, blockSize, info, pBufferSize)
end

function cusparseDbsric02_bufferSizeExt(handle, dirA, mb, nnzb, descrA, bsrSortedVal,
                                        bsrSortedRowPtr, bsrSortedColInd, blockSize, info,
                                        pBufferSize)
    @check @runtime_ccall((:cusparseDbsric02_bufferSizeExt, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{Cdouble}, CuPtr{Cint}, CuPtr{Cint}, Cint, bsric02Info_t,
                  Ptr{Csize_t}),
                 handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr,
                 bsrSortedColInd, blockSize, info, pBufferSize)
end

function cusparseCbsric02_bufferSizeExt(handle, dirA, mb, nnzb, descrA, bsrSortedVal,
                                        bsrSortedRowPtr, bsrSortedColInd, blockSize, info,
                                        pBufferSize)
    @check @runtime_ccall((:cusparseCbsric02_bufferSizeExt, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{cuComplex}, CuPtr{Cint}, CuPtr{Cint}, Cint, bsric02Info_t,
                  Ptr{Csize_t}),
                 handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr,
                 bsrSortedColInd, blockSize, info, pBufferSize)
end

function cusparseZbsric02_bufferSizeExt(handle, dirA, mb, nnzb, descrA, bsrSortedVal,
                                        bsrSortedRowPtr, bsrSortedColInd, blockSize, info,
                                        pBufferSize)
    @check @runtime_ccall((:cusparseZbsric02_bufferSizeExt, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{cuDoubleComplex}, CuPtr{Cint}, CuPtr{Cint}, Cint, bsric02Info_t,
                  Ptr{Csize_t}),
                 handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr,
                 bsrSortedColInd, blockSize, info, pBufferSize)
end

function cusparseSbsric02_analysis(handle, dirA, mb, nnzb, descrA, bsrSortedVal,
                                   bsrSortedRowPtr, bsrSortedColInd, blockDim, info,
                                   policy, pInputBuffer)
    @check @runtime_ccall((:cusparseSbsric02_analysis, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{Cfloat}, CuPtr{Cint}, CuPtr{Cint}, Cint, bsric02Info_t,
                  cusparseSolvePolicy_t, CuPtr{Cvoid}),
                 handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr,
                 bsrSortedColInd, blockDim, info, policy, pInputBuffer)
end

function cusparseDbsric02_analysis(handle, dirA, mb, nnzb, descrA, bsrSortedVal,
                                   bsrSortedRowPtr, bsrSortedColInd, blockDim, info,
                                   policy, pInputBuffer)
    @check @runtime_ccall((:cusparseDbsric02_analysis, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{Cdouble}, CuPtr{Cint}, CuPtr{Cint}, Cint, bsric02Info_t,
                  cusparseSolvePolicy_t, CuPtr{Cvoid}),
                 handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr,
                 bsrSortedColInd, blockDim, info, policy, pInputBuffer)
end

function cusparseCbsric02_analysis(handle, dirA, mb, nnzb, descrA, bsrSortedVal,
                                   bsrSortedRowPtr, bsrSortedColInd, blockDim, info,
                                   policy, pInputBuffer)
    @check @runtime_ccall((:cusparseCbsric02_analysis, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{cuComplex}, CuPtr{Cint}, CuPtr{Cint}, Cint, bsric02Info_t,
                  cusparseSolvePolicy_t, CuPtr{Cvoid}),
                 handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr,
                 bsrSortedColInd, blockDim, info, policy, pInputBuffer)
end

function cusparseZbsric02_analysis(handle, dirA, mb, nnzb, descrA, bsrSortedVal,
                                   bsrSortedRowPtr, bsrSortedColInd, blockDim, info,
                                   policy, pInputBuffer)
    @check @runtime_ccall((:cusparseZbsric02_analysis, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{cuDoubleComplex}, CuPtr{Cint}, CuPtr{Cint}, Cint, bsric02Info_t,
                  cusparseSolvePolicy_t, CuPtr{Cvoid}),
                 handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr,
                 bsrSortedColInd, blockDim, info, policy, pInputBuffer)
end

function cusparseSbsric02(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr,
                          bsrSortedColInd, blockDim, info, policy, pBuffer)
    @check @runtime_ccall((:cusparseSbsric02, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{Cfloat}, CuPtr{Cint}, CuPtr{Cint}, Cint, bsric02Info_t,
                  cusparseSolvePolicy_t, CuPtr{Cvoid}),
                 handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr,
                 bsrSortedColInd, blockDim, info, policy, pBuffer)
end

function cusparseDbsric02(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr,
                          bsrSortedColInd, blockDim, info, policy, pBuffer)
    @check @runtime_ccall((:cusparseDbsric02, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{Cdouble}, CuPtr{Cint}, CuPtr{Cint}, Cint, bsric02Info_t,
                  cusparseSolvePolicy_t, CuPtr{Cvoid}),
                 handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr,
                 bsrSortedColInd, blockDim, info, policy, pBuffer)
end

function cusparseCbsric02(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr,
                          bsrSortedColInd, blockDim, info, policy, pBuffer)
    @check @runtime_ccall((:cusparseCbsric02, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{cuComplex}, CuPtr{Cint}, CuPtr{Cint}, Cint, bsric02Info_t,
                  cusparseSolvePolicy_t, CuPtr{Cvoid}),
                 handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr,
                 bsrSortedColInd, blockDim, info, policy, pBuffer)
end

function cusparseZbsric02(handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr,
                          bsrSortedColInd, blockDim, info, policy, pBuffer)
    @check @runtime_ccall((:cusparseZbsric02, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{cuDoubleComplex}, CuPtr{Cint}, CuPtr{Cint}, Cint, bsric02Info_t,
                  cusparseSolvePolicy_t, CuPtr{Cvoid}),
                 handle, dirA, mb, nnzb, descrA, bsrSortedVal, bsrSortedRowPtr,
                 bsrSortedColInd, blockDim, info, policy, pBuffer)
end

function cusparseSgtsv(handle, m, n, dl, d, du, B, ldb)
    @check @runtime_ccall((:cusparseSgtsv, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, CuPtr{Cfloat}, CuPtr{Cfloat},
                  CuPtr{Cfloat}, CuPtr{Cfloat}, Cint),
                 handle, m, n, dl, d, du, B, ldb)
end

function cusparseDgtsv(handle, m, n, dl, d, du, B, ldb)
    @check @runtime_ccall((:cusparseDgtsv, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, CuPtr{Cdouble}, CuPtr{Cdouble},
                  CuPtr{Cdouble}, CuPtr{Cdouble}, Cint),
                 handle, m, n, dl, d, du, B, ldb)
end

function cusparseCgtsv(handle, m, n, dl, d, du, B, ldb)
    @check @runtime_ccall((:cusparseCgtsv, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, CuPtr{cuComplex}, CuPtr{cuComplex},
                  CuPtr{cuComplex}, CuPtr{cuComplex}, Cint),
                 handle, m, n, dl, d, du, B, ldb)
end

function cusparseZgtsv(handle, m, n, dl, d, du, B, ldb)
    @check @runtime_ccall((:cusparseZgtsv, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, CuPtr{cuDoubleComplex},
                  CuPtr{cuDoubleComplex}, CuPtr{cuDoubleComplex}, CuPtr{cuDoubleComplex},
                  Cint),
                 handle, m, n, dl, d, du, B, ldb)
end

function cusparseSgtsv2_bufferSizeExt(handle, m, n, dl, d, du, B, ldb, bufferSizeInBytes)
    @check @runtime_ccall((:cusparseSgtsv2_bufferSizeExt, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, CuPtr{Cfloat}, CuPtr{Cfloat},
                  CuPtr{Cfloat}, CuPtr{Cfloat}, Cint, Ptr{Csize_t}),
                 handle, m, n, dl, d, du, B, ldb, bufferSizeInBytes)
end

function cusparseDgtsv2_bufferSizeExt(handle, m, n, dl, d, du, B, ldb, bufferSizeInBytes)
    @check @runtime_ccall((:cusparseDgtsv2_bufferSizeExt, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, CuPtr{Cdouble}, CuPtr{Cdouble},
                  CuPtr{Cdouble}, CuPtr{Cdouble}, Cint, Ptr{Csize_t}),
                 handle, m, n, dl, d, du, B, ldb, bufferSizeInBytes)
end

function cusparseCgtsv2_bufferSizeExt(handle, m, n, dl, d, du, B, ldb, bufferSizeInBytes)
    @check @runtime_ccall((:cusparseCgtsv2_bufferSizeExt, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, CuPtr{cuComplex}, CuPtr{cuComplex},
                  CuPtr{cuComplex}, CuPtr{cuComplex}, Cint, Ptr{Csize_t}),
                 handle, m, n, dl, d, du, B, ldb, bufferSizeInBytes)
end

function cusparseZgtsv2_bufferSizeExt(handle, m, n, dl, d, du, B, ldb, bufferSizeInBytes)
    @check @runtime_ccall((:cusparseZgtsv2_bufferSizeExt, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, CuPtr{cuDoubleComplex},
                  CuPtr{cuDoubleComplex}, CuPtr{cuDoubleComplex}, CuPtr{cuDoubleComplex},
                  Cint, Ptr{Csize_t}),
                 handle, m, n, dl, d, du, B, ldb, bufferSizeInBytes)
end

function cusparseSgtsv2(handle, m, n, dl, d, du, B, ldb, pBuffer)
    @check @runtime_ccall((:cusparseSgtsv2, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, CuPtr{Cfloat}, CuPtr{Cfloat},
                  CuPtr{Cfloat}, CuPtr{Cfloat}, Cint, CuPtr{Cvoid}),
                 handle, m, n, dl, d, du, B, ldb, pBuffer)
end

function cusparseDgtsv2(handle, m, n, dl, d, du, B, ldb, pBuffer)
    @check @runtime_ccall((:cusparseDgtsv2, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, CuPtr{Cdouble}, CuPtr{Cdouble},
                  CuPtr{Cdouble}, CuPtr{Cdouble}, Cint, CuPtr{Cvoid}),
                 handle, m, n, dl, d, du, B, ldb, pBuffer)
end

function cusparseCgtsv2(handle, m, n, dl, d, du, B, ldb, pBuffer)
    @check @runtime_ccall((:cusparseCgtsv2, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, CuPtr{cuComplex}, CuPtr{cuComplex},
                  CuPtr{cuComplex}, CuPtr{cuComplex}, Cint, CuPtr{Cvoid}),
                 handle, m, n, dl, d, du, B, ldb, pBuffer)
end

function cusparseZgtsv2(handle, m, n, dl, d, du, B, ldb, pBuffer)
    @check @runtime_ccall((:cusparseZgtsv2, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, CuPtr{cuDoubleComplex},
                  CuPtr{cuDoubleComplex}, CuPtr{cuDoubleComplex}, CuPtr{cuDoubleComplex},
                  Cint, CuPtr{Cvoid}),
                 handle, m, n, dl, d, du, B, ldb, pBuffer)
end

function cusparseSgtsv_nopivot(handle, m, n, dl, d, du, B, ldb)
    @check @runtime_ccall((:cusparseSgtsv_nopivot, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, CuPtr{Cfloat}, CuPtr{Cfloat},
                  CuPtr{Cfloat}, CuPtr{Cfloat}, Cint),
                 handle, m, n, dl, d, du, B, ldb)
end

function cusparseDgtsv_nopivot(handle, m, n, dl, d, du, B, ldb)
    @check @runtime_ccall((:cusparseDgtsv_nopivot, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, CuPtr{Cdouble}, CuPtr{Cdouble},
                  CuPtr{Cdouble}, CuPtr{Cdouble}, Cint),
                 handle, m, n, dl, d, du, B, ldb)
end

function cusparseCgtsv_nopivot(handle, m, n, dl, d, du, B, ldb)
    @check @runtime_ccall((:cusparseCgtsv_nopivot, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, CuPtr{cuComplex}, CuPtr{cuComplex},
                  CuPtr{cuComplex}, CuPtr{cuComplex}, Cint),
                 handle, m, n, dl, d, du, B, ldb)
end

function cusparseZgtsv_nopivot(handle, m, n, dl, d, du, B, ldb)
    @check @runtime_ccall((:cusparseZgtsv_nopivot, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, CuPtr{cuDoubleComplex},
                  CuPtr{cuDoubleComplex}, CuPtr{cuDoubleComplex}, CuPtr{cuDoubleComplex},
                  Cint),
                 handle, m, n, dl, d, du, B, ldb)
end

function cusparseSgtsv2_nopivot_bufferSizeExt(handle, m, n, dl, d, du, B, ldb,
                                              bufferSizeInBytes)
    @check @runtime_ccall((:cusparseSgtsv2_nopivot_bufferSizeExt, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, CuPtr{Cfloat}, CuPtr{Cfloat},
                  CuPtr{Cfloat}, CuPtr{Cfloat}, Cint, Ptr{Csize_t}),
                 handle, m, n, dl, d, du, B, ldb, bufferSizeInBytes)
end

function cusparseDgtsv2_nopivot_bufferSizeExt(handle, m, n, dl, d, du, B, ldb,
                                              bufferSizeInBytes)
    @check @runtime_ccall((:cusparseDgtsv2_nopivot_bufferSizeExt, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, CuPtr{Cdouble}, CuPtr{Cdouble},
                  CuPtr{Cdouble}, CuPtr{Cdouble}, Cint, Ptr{Csize_t}),
                 handle, m, n, dl, d, du, B, ldb, bufferSizeInBytes)
end

function cusparseCgtsv2_nopivot_bufferSizeExt(handle, m, n, dl, d, du, B, ldb,
                                              bufferSizeInBytes)
    @check @runtime_ccall((:cusparseCgtsv2_nopivot_bufferSizeExt, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, CuPtr{cuComplex}, CuPtr{cuComplex},
                  CuPtr{cuComplex}, CuPtr{cuComplex}, Cint, Ptr{Csize_t}),
                 handle, m, n, dl, d, du, B, ldb, bufferSizeInBytes)
end

function cusparseZgtsv2_nopivot_bufferSizeExt(handle, m, n, dl, d, du, B, ldb,
                                              bufferSizeInBytes)
    @check @runtime_ccall((:cusparseZgtsv2_nopivot_bufferSizeExt, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, CuPtr{cuDoubleComplex},
                  CuPtr{cuDoubleComplex}, CuPtr{cuDoubleComplex}, CuPtr{cuDoubleComplex},
                  Cint, Ptr{Csize_t}),
                 handle, m, n, dl, d, du, B, ldb, bufferSizeInBytes)
end

function cusparseSgtsv2_nopivot(handle, m, n, dl, d, du, B, ldb, pBuffer)
    @check @runtime_ccall((:cusparseSgtsv2_nopivot, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, CuPtr{Cfloat}, CuPtr{Cfloat},
                  CuPtr{Cfloat}, CuPtr{Cfloat}, Cint, CuPtr{Cvoid}),
                 handle, m, n, dl, d, du, B, ldb, pBuffer)
end

function cusparseDgtsv2_nopivot(handle, m, n, dl, d, du, B, ldb, pBuffer)
    @check @runtime_ccall((:cusparseDgtsv2_nopivot, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, CuPtr{Cdouble}, CuPtr{Cdouble},
                  CuPtr{Cdouble}, CuPtr{Cdouble}, Cint, CuPtr{Cvoid}),
                 handle, m, n, dl, d, du, B, ldb, pBuffer)
end

function cusparseCgtsv2_nopivot(handle, m, n, dl, d, du, B, ldb, pBuffer)
    @check @runtime_ccall((:cusparseCgtsv2_nopivot, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, CuPtr{cuComplex}, CuPtr{cuComplex},
                  CuPtr{cuComplex}, CuPtr{cuComplex}, Cint, CuPtr{Cvoid}),
                 handle, m, n, dl, d, du, B, ldb, pBuffer)
end

function cusparseZgtsv2_nopivot(handle, m, n, dl, d, du, B, ldb, pBuffer)
    @check @runtime_ccall((:cusparseZgtsv2_nopivot, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, CuPtr{cuDoubleComplex},
                  CuPtr{cuDoubleComplex}, CuPtr{cuDoubleComplex}, CuPtr{cuDoubleComplex},
                  Cint, CuPtr{Cvoid}),
                 handle, m, n, dl, d, du, B, ldb, pBuffer)
end

function cusparseSgtsvStridedBatch(handle, m, dl, d, du, x, batchCount, batchStride)
    @check @runtime_ccall((:cusparseSgtsvStridedBatch, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, CuPtr{Cfloat}, CuPtr{Cfloat}, CuPtr{Cfloat},
                  CuPtr{Cfloat}, Cint, Cint),
                 handle, m, dl, d, du, x, batchCount, batchStride)
end

function cusparseDgtsvStridedBatch(handle, m, dl, d, du, x, batchCount, batchStride)
    @check @runtime_ccall((:cusparseDgtsvStridedBatch, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, CuPtr{Cdouble}, CuPtr{Cdouble}, CuPtr{Cdouble},
                  CuPtr{Cdouble}, Cint, Cint),
                 handle, m, dl, d, du, x, batchCount, batchStride)
end

function cusparseCgtsvStridedBatch(handle, m, dl, d, du, x, batchCount, batchStride)
    @check @runtime_ccall((:cusparseCgtsvStridedBatch, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, CuPtr{cuComplex}, CuPtr{cuComplex},
                  CuPtr{cuComplex}, CuPtr{cuComplex}, Cint, Cint),
                 handle, m, dl, d, du, x, batchCount, batchStride)
end

function cusparseZgtsvStridedBatch(handle, m, dl, d, du, x, batchCount, batchStride)
    @check @runtime_ccall((:cusparseZgtsvStridedBatch, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, CuPtr{cuDoubleComplex}, CuPtr{cuDoubleComplex},
                  CuPtr{cuDoubleComplex}, CuPtr{cuDoubleComplex}, Cint, Cint),
                 handle, m, dl, d, du, x, batchCount, batchStride)
end

function cusparseSgtsv2StridedBatch_bufferSizeExt(handle, m, dl, d, du, x, batchCount,
                                                  batchStride, bufferSizeInBytes)
    @check @runtime_ccall((:cusparseSgtsv2StridedBatch_bufferSizeExt, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, CuPtr{Cfloat}, CuPtr{Cfloat}, CuPtr{Cfloat},
                  CuPtr{Cfloat}, Cint, Cint, Ptr{Csize_t}),
                 handle, m, dl, d, du, x, batchCount, batchStride, bufferSizeInBytes)
end

function cusparseDgtsv2StridedBatch_bufferSizeExt(handle, m, dl, d, du, x, batchCount,
                                                  batchStride, bufferSizeInBytes)
    @check @runtime_ccall((:cusparseDgtsv2StridedBatch_bufferSizeExt, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, CuPtr{Cdouble}, CuPtr{Cdouble}, CuPtr{Cdouble},
                  CuPtr{Cdouble}, Cint, Cint, Ptr{Csize_t}),
                 handle, m, dl, d, du, x, batchCount, batchStride, bufferSizeInBytes)
end

function cusparseCgtsv2StridedBatch_bufferSizeExt(handle, m, dl, d, du, x, batchCount,
                                                  batchStride, bufferSizeInBytes)
    @check @runtime_ccall((:cusparseCgtsv2StridedBatch_bufferSizeExt, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, CuPtr{cuComplex}, CuPtr{cuComplex},
                  CuPtr{cuComplex}, CuPtr{cuComplex}, Cint, Cint, Ptr{Csize_t}),
                 handle, m, dl, d, du, x, batchCount, batchStride, bufferSizeInBytes)
end

function cusparseZgtsv2StridedBatch_bufferSizeExt(handle, m, dl, d, du, x, batchCount,
                                                  batchStride, bufferSizeInBytes)
    @check @runtime_ccall((:cusparseZgtsv2StridedBatch_bufferSizeExt, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, CuPtr{cuDoubleComplex}, CuPtr{cuDoubleComplex},
                  CuPtr{cuDoubleComplex}, CuPtr{cuDoubleComplex}, Cint, Cint, Ptr{Csize_t}),
                 handle, m, dl, d, du, x, batchCount, batchStride, bufferSizeInBytes)
end

function cusparseSgtsv2StridedBatch(handle, m, dl, d, du, x, batchCount, batchStride,
                                    pBuffer)
    @check @runtime_ccall((:cusparseSgtsv2StridedBatch, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, CuPtr{Cfloat}, CuPtr{Cfloat}, CuPtr{Cfloat},
                  CuPtr{Cfloat}, Cint, Cint, CuPtr{Cvoid}),
                 handle, m, dl, d, du, x, batchCount, batchStride, pBuffer)
end

function cusparseDgtsv2StridedBatch(handle, m, dl, d, du, x, batchCount, batchStride,
                                    pBuffer)
    @check @runtime_ccall((:cusparseDgtsv2StridedBatch, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, CuPtr{Cdouble}, CuPtr{Cdouble}, CuPtr{Cdouble},
                  CuPtr{Cdouble}, Cint, Cint, CuPtr{Cvoid}),
                 handle, m, dl, d, du, x, batchCount, batchStride, pBuffer)
end

function cusparseCgtsv2StridedBatch(handle, m, dl, d, du, x, batchCount, batchStride,
                                    pBuffer)
    @check @runtime_ccall((:cusparseCgtsv2StridedBatch, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, CuPtr{cuComplex}, CuPtr{cuComplex},
                  CuPtr{cuComplex}, CuPtr{cuComplex}, Cint, Cint, CuPtr{Cvoid}),
                 handle, m, dl, d, du, x, batchCount, batchStride, pBuffer)
end

function cusparseZgtsv2StridedBatch(handle, m, dl, d, du, x, batchCount, batchStride,
                                    pBuffer)
    @check @runtime_ccall((:cusparseZgtsv2StridedBatch, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, CuPtr{cuDoubleComplex}, CuPtr{cuDoubleComplex},
                  CuPtr{cuDoubleComplex}, CuPtr{cuDoubleComplex}, Cint, Cint, CuPtr{Cvoid}),
                 handle, m, dl, d, du, x, batchCount, batchStride, pBuffer)
end

function cusparseSgtsvInterleavedBatch_bufferSizeExt(handle, algo, m, dl, d, du, x,
                                                     batchCount, pBufferSizeInBytes)
    @check @runtime_ccall((:cusparseSgtsvInterleavedBatch_bufferSizeExt, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, CuPtr{Cfloat}, CuPtr{Cfloat},
                  CuPtr{Cfloat}, CuPtr{Cfloat}, Cint, Ptr{Csize_t}),
                 handle, algo, m, dl, d, du, x, batchCount, pBufferSizeInBytes)
end

function cusparseDgtsvInterleavedBatch_bufferSizeExt(handle, algo, m, dl, d, du, x,
                                                     batchCount, pBufferSizeInBytes)
    @check @runtime_ccall((:cusparseDgtsvInterleavedBatch_bufferSizeExt, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, CuPtr{Cdouble}, CuPtr{Cdouble},
                  CuPtr{Cdouble}, CuPtr{Cdouble}, Cint, Ptr{Csize_t}),
                 handle, algo, m, dl, d, du, x, batchCount, pBufferSizeInBytes)
end

function cusparseCgtsvInterleavedBatch_bufferSizeExt(handle, algo, m, dl, d, du, x,
                                                     batchCount, pBufferSizeInBytes)
    @check @runtime_ccall((:cusparseCgtsvInterleavedBatch_bufferSizeExt, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, CuPtr{cuComplex}, CuPtr{cuComplex},
                  CuPtr{cuComplex}, CuPtr{cuComplex}, Cint, Ptr{Csize_t}),
                 handle, algo, m, dl, d, du, x, batchCount, pBufferSizeInBytes)
end

function cusparseZgtsvInterleavedBatch_bufferSizeExt(handle, algo, m, dl, d, du, x,
                                                     batchCount, pBufferSizeInBytes)
    @check @runtime_ccall((:cusparseZgtsvInterleavedBatch_bufferSizeExt, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, CuPtr{cuDoubleComplex},
                  CuPtr{cuDoubleComplex}, CuPtr{cuDoubleComplex}, CuPtr{cuDoubleComplex},
                  Cint, Ptr{Csize_t}),
                 handle, algo, m, dl, d, du, x, batchCount, pBufferSizeInBytes)
end

function cusparseSgtsvInterleavedBatch(handle, algo, m, dl, d, du, x, batchCount, pBuffer)
    @check @runtime_ccall((:cusparseSgtsvInterleavedBatch, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, CuPtr{Cfloat}, CuPtr{Cfloat},
                  CuPtr{Cfloat}, CuPtr{Cfloat}, Cint, CuPtr{Cvoid}),
                 handle, algo, m, dl, d, du, x, batchCount, pBuffer)
end

function cusparseDgtsvInterleavedBatch(handle, algo, m, dl, d, du, x, batchCount, pBuffer)
    @check @runtime_ccall((:cusparseDgtsvInterleavedBatch, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, CuPtr{Cdouble}, CuPtr{Cdouble},
                  CuPtr{Cdouble}, CuPtr{Cdouble}, Cint, CuPtr{Cvoid}),
                 handle, algo, m, dl, d, du, x, batchCount, pBuffer)
end

function cusparseCgtsvInterleavedBatch(handle, algo, m, dl, d, du, x, batchCount, pBuffer)
    @check @runtime_ccall((:cusparseCgtsvInterleavedBatch, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, CuPtr{cuComplex}, CuPtr{cuComplex},
                  CuPtr{cuComplex}, CuPtr{cuComplex}, Cint, CuPtr{Cvoid}),
                 handle, algo, m, dl, d, du, x, batchCount, pBuffer)
end

function cusparseZgtsvInterleavedBatch(handle, algo, m, dl, d, du, x, batchCount, pBuffer)
    @check @runtime_ccall((:cusparseZgtsvInterleavedBatch, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, CuPtr{cuDoubleComplex},
                  CuPtr{cuDoubleComplex}, CuPtr{cuDoubleComplex}, CuPtr{cuDoubleComplex},
                  Cint, CuPtr{Cvoid}),
                 handle, algo, m, dl, d, du, x, batchCount, pBuffer)
end

function cusparseSgpsvInterleavedBatch_bufferSizeExt(handle, algo, m, ds, dl, d, du, dw, x,
                                                     batchCount, pBufferSizeInBytes)
    @check @runtime_ccall((:cusparseSgpsvInterleavedBatch_bufferSizeExt, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, CuPtr{Cfloat}, CuPtr{Cfloat},
                  CuPtr{Cfloat}, CuPtr{Cfloat}, CuPtr{Cfloat}, CuPtr{Cfloat}, Cint,
                  Ptr{Csize_t}),
                 handle, algo, m, ds, dl, d, du, dw, x, batchCount, pBufferSizeInBytes)
end

function cusparseDgpsvInterleavedBatch_bufferSizeExt(handle, algo, m, ds, dl, d, du, dw, x,
                                                     batchCount, pBufferSizeInBytes)
    @check @runtime_ccall((:cusparseDgpsvInterleavedBatch_bufferSizeExt, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, CuPtr{Cdouble}, CuPtr{Cdouble},
                  CuPtr{Cdouble}, CuPtr{Cdouble}, CuPtr{Cdouble}, CuPtr{Cdouble}, Cint,
                  Ptr{Csize_t}),
                 handle, algo, m, ds, dl, d, du, dw, x, batchCount, pBufferSizeInBytes)
end

function cusparseCgpsvInterleavedBatch_bufferSizeExt(handle, algo, m, ds, dl, d, du, dw, x,
                                                     batchCount, pBufferSizeInBytes)
    @check @runtime_ccall((:cusparseCgpsvInterleavedBatch_bufferSizeExt, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, CuPtr{cuComplex}, CuPtr{cuComplex},
                  CuPtr{cuComplex}, CuPtr{cuComplex}, CuPtr{cuComplex}, CuPtr{cuComplex},
                  Cint, Ptr{Csize_t}),
                 handle, algo, m, ds, dl, d, du, dw, x, batchCount, pBufferSizeInBytes)
end

function cusparseZgpsvInterleavedBatch_bufferSizeExt(handle, algo, m, ds, dl, d, du, dw, x,
                                                     batchCount, pBufferSizeInBytes)
    @check @runtime_ccall((:cusparseZgpsvInterleavedBatch_bufferSizeExt, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, CuPtr{cuDoubleComplex},
                  CuPtr{cuDoubleComplex}, CuPtr{cuDoubleComplex}, CuPtr{cuDoubleComplex},
                  CuPtr{cuDoubleComplex}, CuPtr{cuDoubleComplex}, Cint, Ptr{Csize_t}),
                 handle, algo, m, ds, dl, d, du, dw, x, batchCount, pBufferSizeInBytes)
end

function cusparseSgpsvInterleavedBatch(handle, algo, m, ds, dl, d, du, dw, x, batchCount,
                                       pBuffer)
    @check @runtime_ccall((:cusparseSgpsvInterleavedBatch, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, CuPtr{Cfloat}, CuPtr{Cfloat},
                  CuPtr{Cfloat}, CuPtr{Cfloat}, CuPtr{Cfloat}, CuPtr{Cfloat}, Cint,
                  CuPtr{Cvoid}),
                 handle, algo, m, ds, dl, d, du, dw, x, batchCount, pBuffer)
end

function cusparseDgpsvInterleavedBatch(handle, algo, m, ds, dl, d, du, dw, x, batchCount,
                                       pBuffer)
    @check @runtime_ccall((:cusparseDgpsvInterleavedBatch, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, CuPtr{Cdouble}, CuPtr{Cdouble},
                  CuPtr{Cdouble}, CuPtr{Cdouble}, CuPtr{Cdouble}, CuPtr{Cdouble}, Cint,
                  CuPtr{Cvoid}),
                 handle, algo, m, ds, dl, d, du, dw, x, batchCount, pBuffer)
end

function cusparseCgpsvInterleavedBatch(handle, algo, m, ds, dl, d, du, dw, x, batchCount,
                                       pBuffer)
    @check @runtime_ccall((:cusparseCgpsvInterleavedBatch, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, CuPtr{cuComplex}, CuPtr{cuComplex},
                  CuPtr{cuComplex}, CuPtr{cuComplex}, CuPtr{cuComplex}, CuPtr{cuComplex},
                  Cint, CuPtr{Cvoid}),
                 handle, algo, m, ds, dl, d, du, dw, x, batchCount, pBuffer)
end

function cusparseZgpsvInterleavedBatch(handle, algo, m, ds, dl, d, du, dw, x, batchCount,
                                       pBuffer)
    @check @runtime_ccall((:cusparseZgpsvInterleavedBatch, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, CuPtr{cuDoubleComplex},
                  CuPtr{cuDoubleComplex}, CuPtr{cuDoubleComplex}, CuPtr{cuDoubleComplex},
                  CuPtr{cuDoubleComplex}, CuPtr{cuDoubleComplex}, Cint, CuPtr{Cvoid}),
                 handle, algo, m, ds, dl, d, du, dw, x, batchCount, pBuffer)
end

function cusparseXcsrgemmNnz(handle, transA, transB, m, n, k, descrA, nnzA,
                             csrSortedRowPtrA, csrSortedColIndA, descrB, nnzB,
                             csrSortedRowPtrB, csrSortedColIndB, descrC, csrSortedRowPtrC,
                             nnzTotalDevHostPtr)
    @check @runtime_ccall((:cusparseXcsrgemmNnz, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, Cint, Cint,
                  Cint, cusparseMatDescr_t, Cint, CuPtr{Cint}, CuPtr{Cint},
                  cusparseMatDescr_t, Cint, CuPtr{Cint}, CuPtr{Cint}, cusparseMatDescr_t,
                  CuPtr{Cint}, PtrOrCuPtr{Cint}),
                 handle, transA, transB, m, n, k, descrA, nnzA, csrSortedRowPtrA,
                 csrSortedColIndA, descrB, nnzB, csrSortedRowPtrB, csrSortedColIndB,
                 descrC, csrSortedRowPtrC, nnzTotalDevHostPtr)
end

function cusparseScsrgemm(handle, transA, transB, m, n, k, descrA, nnzA, csrSortedValA,
                          csrSortedRowPtrA, csrSortedColIndA, descrB, nnzB, csrSortedValB,
                          csrSortedRowPtrB, csrSortedColIndB, descrC, csrSortedValC,
                          csrSortedRowPtrC, csrSortedColIndC)
    @check @runtime_ccall((:cusparseScsrgemm, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, Cint, Cint,
                  Cint, cusparseMatDescr_t, Cint, CuPtr{Cfloat}, CuPtr{Cint}, CuPtr{Cint},
                  cusparseMatDescr_t, Cint, CuPtr{Cfloat}, CuPtr{Cint}, CuPtr{Cint},
                  cusparseMatDescr_t, CuPtr{Cfloat}, CuPtr{Cint}, CuPtr{Cint}),
                 handle, transA, transB, m, n, k, descrA, nnzA, csrSortedValA,
                 csrSortedRowPtrA, csrSortedColIndA, descrB, nnzB, csrSortedValB,
                 csrSortedRowPtrB, csrSortedColIndB, descrC, csrSortedValC,
                 csrSortedRowPtrC, csrSortedColIndC)
end

function cusparseDcsrgemm(handle, transA, transB, m, n, k, descrA, nnzA, csrSortedValA,
                          csrSortedRowPtrA, csrSortedColIndA, descrB, nnzB, csrSortedValB,
                          csrSortedRowPtrB, csrSortedColIndB, descrC, csrSortedValC,
                          csrSortedRowPtrC, csrSortedColIndC)
    @check @runtime_ccall((:cusparseDcsrgemm, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, Cint, Cint,
                  Cint, cusparseMatDescr_t, Cint, CuPtr{Cdouble}, CuPtr{Cint}, CuPtr{Cint},
                  cusparseMatDescr_t, Cint, CuPtr{Cdouble}, CuPtr{Cint}, CuPtr{Cint},
                  cusparseMatDescr_t, CuPtr{Cdouble}, CuPtr{Cint}, CuPtr{Cint}),
                 handle, transA, transB, m, n, k, descrA, nnzA, csrSortedValA,
                 csrSortedRowPtrA, csrSortedColIndA, descrB, nnzB, csrSortedValB,
                 csrSortedRowPtrB, csrSortedColIndB, descrC, csrSortedValC,
                 csrSortedRowPtrC, csrSortedColIndC)
end

function cusparseCcsrgemm(handle, transA, transB, m, n, k, descrA, nnzA, csrSortedValA,
                          csrSortedRowPtrA, csrSortedColIndA, descrB, nnzB, csrSortedValB,
                          csrSortedRowPtrB, csrSortedColIndB, descrC, csrSortedValC,
                          csrSortedRowPtrC, csrSortedColIndC)
    @check @runtime_ccall((:cusparseCcsrgemm, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, Cint, Cint,
                  Cint, cusparseMatDescr_t, Cint, CuPtr{cuComplex}, CuPtr{Cint},
                  CuPtr{Cint}, cusparseMatDescr_t, Cint, CuPtr{cuComplex}, CuPtr{Cint},
                  CuPtr{Cint}, cusparseMatDescr_t, CuPtr{cuComplex}, CuPtr{Cint},
                  CuPtr{Cint}),
                 handle, transA, transB, m, n, k, descrA, nnzA, csrSortedValA,
                 csrSortedRowPtrA, csrSortedColIndA, descrB, nnzB, csrSortedValB,
                 csrSortedRowPtrB, csrSortedColIndB, descrC, csrSortedValC,
                 csrSortedRowPtrC, csrSortedColIndC)
end

function cusparseZcsrgemm(handle, transA, transB, m, n, k, descrA, nnzA, csrSortedValA,
                          csrSortedRowPtrA, csrSortedColIndA, descrB, nnzB, csrSortedValB,
                          csrSortedRowPtrB, csrSortedColIndB, descrC, csrSortedValC,
                          csrSortedRowPtrC, csrSortedColIndC)
    @check @runtime_ccall((:cusparseZcsrgemm, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, Cint, Cint,
                  Cint, cusparseMatDescr_t, Cint, CuPtr{cuDoubleComplex}, CuPtr{Cint},
                  CuPtr{Cint}, cusparseMatDescr_t, Cint, CuPtr{cuDoubleComplex},
                  CuPtr{Cint}, CuPtr{Cint}, cusparseMatDescr_t, CuPtr{cuDoubleComplex},
                  CuPtr{Cint}, CuPtr{Cint}),
                 handle, transA, transB, m, n, k, descrA, nnzA, csrSortedValA,
                 csrSortedRowPtrA, csrSortedColIndA, descrB, nnzB, csrSortedValB,
                 csrSortedRowPtrB, csrSortedColIndB, descrC, csrSortedValC,
                 csrSortedRowPtrC, csrSortedColIndC)
end

function cusparseCreateCsrgemm2Info(info)
    @check @runtime_ccall((:cusparseCreateCsrgemm2Info, libcusparse()), cusparseStatus_t,
                 (Ptr{csrgemm2Info_t},),
                 info)
end

function cusparseDestroyCsrgemm2Info(info)
    @check @runtime_ccall((:cusparseDestroyCsrgemm2Info, libcusparse()), cusparseStatus_t,
                 (csrgemm2Info_t,),
                 info)
end

function cusparseScsrgemm2_bufferSizeExt(handle, m, n, k, alpha, descrA, nnzA,
                                         csrSortedRowPtrA, csrSortedColIndA, descrB, nnzB,
                                         csrSortedRowPtrB, csrSortedColIndB, beta, descrD,
                                         nnzD, csrSortedRowPtrD, csrSortedColIndD, info,
                                         pBufferSizeInBytes)
    @check @runtime_ccall((:cusparseScsrgemm2_bufferSizeExt, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, Cint, Ptr{Cfloat}, cusparseMatDescr_t,
                  Cint, CuPtr{Cint}, CuPtr{Cint}, cusparseMatDescr_t, Cint, CuPtr{Cint},
                  CuPtr{Cint}, Ptr{Cfloat}, cusparseMatDescr_t, Cint, CuPtr{Cint},
                  CuPtr{Cint}, csrgemm2Info_t, Ptr{Csize_t}),
                 handle, m, n, k, alpha, descrA, nnzA, csrSortedRowPtrA, csrSortedColIndA,
                 descrB, nnzB, csrSortedRowPtrB, csrSortedColIndB, beta, descrD, nnzD,
                 csrSortedRowPtrD, csrSortedColIndD, info, pBufferSizeInBytes)
end

function cusparseDcsrgemm2_bufferSizeExt(handle, m, n, k, alpha, descrA, nnzA,
                                         csrSortedRowPtrA, csrSortedColIndA, descrB, nnzB,
                                         csrSortedRowPtrB, csrSortedColIndB, beta, descrD,
                                         nnzD, csrSortedRowPtrD, csrSortedColIndD, info,
                                         pBufferSizeInBytes)
    @check @runtime_ccall((:cusparseDcsrgemm2_bufferSizeExt, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, Cint, Ptr{Cdouble}, cusparseMatDescr_t,
                  Cint, CuPtr{Cint}, CuPtr{Cint}, cusparseMatDescr_t, Cint, CuPtr{Cint},
                  CuPtr{Cint}, Ptr{Cdouble}, cusparseMatDescr_t, Cint, CuPtr{Cint},
                  CuPtr{Cint}, csrgemm2Info_t, Ptr{Csize_t}),
                 handle, m, n, k, alpha, descrA, nnzA, csrSortedRowPtrA, csrSortedColIndA,
                 descrB, nnzB, csrSortedRowPtrB, csrSortedColIndB, beta, descrD, nnzD,
                 csrSortedRowPtrD, csrSortedColIndD, info, pBufferSizeInBytes)
end

function cusparseCcsrgemm2_bufferSizeExt(handle, m, n, k, alpha, descrA, nnzA,
                                         csrSortedRowPtrA, csrSortedColIndA, descrB, nnzB,
                                         csrSortedRowPtrB, csrSortedColIndB, beta, descrD,
                                         nnzD, csrSortedRowPtrD, csrSortedColIndD, info,
                                         pBufferSizeInBytes)
    @check @runtime_ccall((:cusparseCcsrgemm2_bufferSizeExt, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, Cint, Ptr{cuComplex}, cusparseMatDescr_t,
                  Cint, CuPtr{Cint}, CuPtr{Cint}, cusparseMatDescr_t, Cint, CuPtr{Cint},
                  CuPtr{Cint}, Ptr{cuComplex}, cusparseMatDescr_t, Cint, CuPtr{Cint},
                  CuPtr{Cint}, csrgemm2Info_t, Ptr{Csize_t}),
                 handle, m, n, k, alpha, descrA, nnzA, csrSortedRowPtrA, csrSortedColIndA,
                 descrB, nnzB, csrSortedRowPtrB, csrSortedColIndB, beta, descrD, nnzD,
                 csrSortedRowPtrD, csrSortedColIndD, info, pBufferSizeInBytes)
end

function cusparseZcsrgemm2_bufferSizeExt(handle, m, n, k, alpha, descrA, nnzA,
                                         csrSortedRowPtrA, csrSortedColIndA, descrB, nnzB,
                                         csrSortedRowPtrB, csrSortedColIndB, beta, descrD,
                                         nnzD, csrSortedRowPtrD, csrSortedColIndD, info,
                                         pBufferSizeInBytes)
    @check @runtime_ccall((:cusparseZcsrgemm2_bufferSizeExt, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, Cint, Ptr{cuDoubleComplex},
                  cusparseMatDescr_t, Cint, CuPtr{Cint}, CuPtr{Cint}, cusparseMatDescr_t,
                  Cint, CuPtr{Cint}, CuPtr{Cint}, Ptr{cuDoubleComplex}, cusparseMatDescr_t,
                  Cint, CuPtr{Cint}, CuPtr{Cint}, csrgemm2Info_t, Ptr{Csize_t}),
                 handle, m, n, k, alpha, descrA, nnzA, csrSortedRowPtrA, csrSortedColIndA,
                 descrB, nnzB, csrSortedRowPtrB, csrSortedColIndB, beta, descrD, nnzD,
                 csrSortedRowPtrD, csrSortedColIndD, info, pBufferSizeInBytes)
end

function cusparseXcsrgemm2Nnz(handle, m, n, k, descrA, nnzA, csrSortedRowPtrA,
                              csrSortedColIndA, descrB, nnzB, csrSortedRowPtrB,
                              csrSortedColIndB, descrD, nnzD, csrSortedRowPtrD,
                              csrSortedColIndD, descrC, csrSortedRowPtrC,
                              nnzTotalDevHostPtr, info, pBuffer)
    @check @runtime_ccall((:cusparseXcsrgemm2Nnz, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, Cint, cusparseMatDescr_t, Cint,
                  CuPtr{Cint}, CuPtr{Cint}, cusparseMatDescr_t, Cint, CuPtr{Cint},
                  CuPtr{Cint}, cusparseMatDescr_t, Cint, CuPtr{Cint}, CuPtr{Cint},
                  cusparseMatDescr_t, CuPtr{Cint}, PtrOrCuPtr{Cint}, csrgemm2Info_t,
                  CuPtr{Cvoid}),
                 handle, m, n, k, descrA, nnzA, csrSortedRowPtrA, csrSortedColIndA, descrB,
                 nnzB, csrSortedRowPtrB, csrSortedColIndB, descrD, nnzD, csrSortedRowPtrD,
                 csrSortedColIndD, descrC, csrSortedRowPtrC, nnzTotalDevHostPtr, info,
                 pBuffer)
end

function cusparseScsrgemm2(handle, m, n, k, alpha, descrA, nnzA, csrSortedValA,
                           csrSortedRowPtrA, csrSortedColIndA, descrB, nnzB, csrSortedValB,
                           csrSortedRowPtrB, csrSortedColIndB, beta, descrD, nnzD,
                           csrSortedValD, csrSortedRowPtrD, csrSortedColIndD, descrC,
                           csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, info, pBuffer)
    @check @runtime_ccall((:cusparseScsrgemm2, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, Cint, Ptr{Cfloat}, cusparseMatDescr_t,
                  Cint, CuPtr{Cfloat}, CuPtr{Cint}, CuPtr{Cint}, cusparseMatDescr_t, Cint,
                  CuPtr{Cfloat}, CuPtr{Cint}, CuPtr{Cint}, Ptr{Cfloat}, cusparseMatDescr_t,
                  Cint, CuPtr{Cfloat}, CuPtr{Cint}, CuPtr{Cint}, cusparseMatDescr_t,
                  CuPtr{Cfloat}, CuPtr{Cint}, CuPtr{Cint}, csrgemm2Info_t, CuPtr{Cvoid}),
                 handle, m, n, k, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA,
                 csrSortedColIndA, descrB, nnzB, csrSortedValB, csrSortedRowPtrB,
                 csrSortedColIndB, beta, descrD, nnzD, csrSortedValD, csrSortedRowPtrD,
                 csrSortedColIndD, descrC, csrSortedValC, csrSortedRowPtrC,
                 csrSortedColIndC, info, pBuffer)
end

function cusparseDcsrgemm2(handle, m, n, k, alpha, descrA, nnzA, csrSortedValA,
                           csrSortedRowPtrA, csrSortedColIndA, descrB, nnzB, csrSortedValB,
                           csrSortedRowPtrB, csrSortedColIndB, beta, descrD, nnzD,
                           csrSortedValD, csrSortedRowPtrD, csrSortedColIndD, descrC,
                           csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, info, pBuffer)
    @check @runtime_ccall((:cusparseDcsrgemm2, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, Cint, Ptr{Cdouble}, cusparseMatDescr_t,
                  Cint, CuPtr{Cdouble}, CuPtr{Cint}, CuPtr{Cint}, cusparseMatDescr_t, Cint,
                  CuPtr{Cdouble}, CuPtr{Cint}, CuPtr{Cint}, Ptr{Cdouble},
                  cusparseMatDescr_t, Cint, CuPtr{Cdouble}, CuPtr{Cint}, CuPtr{Cint},
                  cusparseMatDescr_t, CuPtr{Cdouble}, CuPtr{Cint}, CuPtr{Cint},
                  csrgemm2Info_t, CuPtr{Cvoid}),
                 handle, m, n, k, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA,
                 csrSortedColIndA, descrB, nnzB, csrSortedValB, csrSortedRowPtrB,
                 csrSortedColIndB, beta, descrD, nnzD, csrSortedValD, csrSortedRowPtrD,
                 csrSortedColIndD, descrC, csrSortedValC, csrSortedRowPtrC,
                 csrSortedColIndC, info, pBuffer)
end

function cusparseCcsrgemm2(handle, m, n, k, alpha, descrA, nnzA, csrSortedValA,
                           csrSortedRowPtrA, csrSortedColIndA, descrB, nnzB, csrSortedValB,
                           csrSortedRowPtrB, csrSortedColIndB, beta, descrD, nnzD,
                           csrSortedValD, csrSortedRowPtrD, csrSortedColIndD, descrC,
                           csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, info, pBuffer)
    @check @runtime_ccall((:cusparseCcsrgemm2, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, Cint, Ptr{cuComplex}, cusparseMatDescr_t,
                  Cint, CuPtr{cuComplex}, CuPtr{Cint}, CuPtr{Cint}, cusparseMatDescr_t,
                  Cint, CuPtr{cuComplex}, CuPtr{Cint}, CuPtr{Cint}, Ptr{cuComplex},
                  cusparseMatDescr_t, Cint, CuPtr{cuComplex}, CuPtr{Cint}, CuPtr{Cint},
                  cusparseMatDescr_t, CuPtr{cuComplex}, CuPtr{Cint}, CuPtr{Cint},
                  csrgemm2Info_t, CuPtr{Cvoid}),
                 handle, m, n, k, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA,
                 csrSortedColIndA, descrB, nnzB, csrSortedValB, csrSortedRowPtrB,
                 csrSortedColIndB, beta, descrD, nnzD, csrSortedValD, csrSortedRowPtrD,
                 csrSortedColIndD, descrC, csrSortedValC, csrSortedRowPtrC,
                 csrSortedColIndC, info, pBuffer)
end

function cusparseZcsrgemm2(handle, m, n, k, alpha, descrA, nnzA, csrSortedValA,
                           csrSortedRowPtrA, csrSortedColIndA, descrB, nnzB, csrSortedValB,
                           csrSortedRowPtrB, csrSortedColIndB, beta, descrD, nnzD,
                           csrSortedValD, csrSortedRowPtrD, csrSortedColIndD, descrC,
                           csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, info, pBuffer)
    @check @runtime_ccall((:cusparseZcsrgemm2, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, Cint, Ptr{cuDoubleComplex},
                  cusparseMatDescr_t, Cint, CuPtr{cuDoubleComplex}, CuPtr{Cint},
                  CuPtr{Cint}, cusparseMatDescr_t, Cint, CuPtr{cuDoubleComplex},
                  CuPtr{Cint}, CuPtr{Cint}, Ptr{cuDoubleComplex}, cusparseMatDescr_t, Cint,
                  CuPtr{cuDoubleComplex}, CuPtr{Cint}, CuPtr{Cint}, cusparseMatDescr_t,
                  CuPtr{cuDoubleComplex}, CuPtr{Cint}, CuPtr{Cint}, csrgemm2Info_t,
                  CuPtr{Cvoid}),
                 handle, m, n, k, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA,
                 csrSortedColIndA, descrB, nnzB, csrSortedValB, csrSortedRowPtrB,
                 csrSortedColIndB, beta, descrD, nnzD, csrSortedValD, csrSortedRowPtrD,
                 csrSortedColIndD, descrC, csrSortedValC, csrSortedRowPtrC,
                 csrSortedColIndC, info, pBuffer)
end

function cusparseXcsrgeamNnz(handle, m, n, descrA, nnzA, csrSortedRowPtrA,
                             csrSortedColIndA, descrB, nnzB, csrSortedRowPtrB,
                             csrSortedColIndB, descrC, csrSortedRowPtrC, nnzTotalDevHostPtr)
    @check @runtime_ccall((:cusparseXcsrgeamNnz, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, cusparseMatDescr_t, Cint, CuPtr{Cint},
                  CuPtr{Cint}, cusparseMatDescr_t, Cint, CuPtr{Cint}, CuPtr{Cint},
                  cusparseMatDescr_t, CuPtr{Cint}, PtrOrCuPtr{Cint}),
                 handle, m, n, descrA, nnzA, csrSortedRowPtrA, csrSortedColIndA, descrB,
                 nnzB, csrSortedRowPtrB, csrSortedColIndB, descrC, csrSortedRowPtrC,
                 nnzTotalDevHostPtr)
end

function cusparseScsrgeam(handle, m, n, alpha, descrA, nnzA, csrSortedValA,
                          csrSortedRowPtrA, csrSortedColIndA, beta, descrB, nnzB,
                          csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, descrC,
                          csrSortedValC, csrSortedRowPtrC, csrSortedColIndC)
    @check @runtime_ccall((:cusparseScsrgeam, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, Ptr{Cfloat}, cusparseMatDescr_t, Cint,
                  CuPtr{Cfloat}, CuPtr{Cint}, CuPtr{Cint}, Ptr{Cfloat}, cusparseMatDescr_t,
                  Cint, CuPtr{Cfloat}, CuPtr{Cint}, CuPtr{Cint}, cusparseMatDescr_t,
                  CuPtr{Cfloat}, CuPtr{Cint}, CuPtr{Cint}),
                 handle, m, n, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA,
                 csrSortedColIndA, beta, descrB, nnzB, csrSortedValB, csrSortedRowPtrB,
                 csrSortedColIndB, descrC, csrSortedValC, csrSortedRowPtrC,
                 csrSortedColIndC)
end

function cusparseDcsrgeam(handle, m, n, alpha, descrA, nnzA, csrSortedValA,
                          csrSortedRowPtrA, csrSortedColIndA, beta, descrB, nnzB,
                          csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, descrC,
                          csrSortedValC, csrSortedRowPtrC, csrSortedColIndC)
    @check @runtime_ccall((:cusparseDcsrgeam, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, Ptr{Cdouble}, cusparseMatDescr_t, Cint,
                  CuPtr{Cdouble}, CuPtr{Cint}, CuPtr{Cint}, Ptr{Cdouble},
                  cusparseMatDescr_t, Cint, CuPtr{Cdouble}, CuPtr{Cint}, CuPtr{Cint},
                  cusparseMatDescr_t, CuPtr{Cdouble}, CuPtr{Cint}, CuPtr{Cint}),
                 handle, m, n, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA,
                 csrSortedColIndA, beta, descrB, nnzB, csrSortedValB, csrSortedRowPtrB,
                 csrSortedColIndB, descrC, csrSortedValC, csrSortedRowPtrC,
                 csrSortedColIndC)
end

function cusparseCcsrgeam(handle, m, n, alpha, descrA, nnzA, csrSortedValA,
                          csrSortedRowPtrA, csrSortedColIndA, beta, descrB, nnzB,
                          csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, descrC,
                          csrSortedValC, csrSortedRowPtrC, csrSortedColIndC)
    @check @runtime_ccall((:cusparseCcsrgeam, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, Ptr{cuComplex}, cusparseMatDescr_t, Cint,
                  CuPtr{cuComplex}, CuPtr{Cint}, CuPtr{Cint}, Ptr{cuComplex},
                  cusparseMatDescr_t, Cint, CuPtr{cuComplex}, CuPtr{Cint}, CuPtr{Cint},
                  cusparseMatDescr_t, CuPtr{cuComplex}, CuPtr{Cint}, CuPtr{Cint}),
                 handle, m, n, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA,
                 csrSortedColIndA, beta, descrB, nnzB, csrSortedValB, csrSortedRowPtrB,
                 csrSortedColIndB, descrC, csrSortedValC, csrSortedRowPtrC,
                 csrSortedColIndC)
end

function cusparseZcsrgeam(handle, m, n, alpha, descrA, nnzA, csrSortedValA,
                          csrSortedRowPtrA, csrSortedColIndA, beta, descrB, nnzB,
                          csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, descrC,
                          csrSortedValC, csrSortedRowPtrC, csrSortedColIndC)
    @check @runtime_ccall((:cusparseZcsrgeam, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, Ptr{cuDoubleComplex}, cusparseMatDescr_t,
                  Cint, CuPtr{cuDoubleComplex}, CuPtr{Cint}, CuPtr{Cint},
                  Ptr{cuDoubleComplex}, cusparseMatDescr_t, Cint, CuPtr{cuDoubleComplex},
                  CuPtr{Cint}, CuPtr{Cint}, cusparseMatDescr_t, CuPtr{cuDoubleComplex},
                  CuPtr{Cint}, CuPtr{Cint}),
                 handle, m, n, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA,
                 csrSortedColIndA, beta, descrB, nnzB, csrSortedValB, csrSortedRowPtrB,
                 csrSortedColIndB, descrC, csrSortedValC, csrSortedRowPtrC,
                 csrSortedColIndC)
end

function cusparseScsrgeam2_bufferSizeExt(handle, m, n, alpha, descrA, nnzA, csrSortedValA,
                                         csrSortedRowPtrA, csrSortedColIndA, beta, descrB,
                                         nnzB, csrSortedValB, csrSortedRowPtrB,
                                         csrSortedColIndB, descrC, csrSortedValC,
                                         csrSortedRowPtrC, csrSortedColIndC,
                                         pBufferSizeInBytes)
    @check @runtime_ccall((:cusparseScsrgeam2_bufferSizeExt, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, Ptr{Cfloat}, cusparseMatDescr_t, Cint,
                  CuPtr{Cfloat}, CuPtr{Cint}, CuPtr{Cint}, Ptr{Cfloat}, cusparseMatDescr_t,
                  Cint, CuPtr{Cfloat}, CuPtr{Cint}, CuPtr{Cint}, cusparseMatDescr_t,
                  CuPtr{Cfloat}, CuPtr{Cint}, CuPtr{Cint}, Ptr{Csize_t}),
                 handle, m, n, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA,
                 csrSortedColIndA, beta, descrB, nnzB, csrSortedValB, csrSortedRowPtrB,
                 csrSortedColIndB, descrC, csrSortedValC, csrSortedRowPtrC,
                 csrSortedColIndC, pBufferSizeInBytes)
end

function cusparseDcsrgeam2_bufferSizeExt(handle, m, n, alpha, descrA, nnzA, csrSortedValA,
                                         csrSortedRowPtrA, csrSortedColIndA, beta, descrB,
                                         nnzB, csrSortedValB, csrSortedRowPtrB,
                                         csrSortedColIndB, descrC, csrSortedValC,
                                         csrSortedRowPtrC, csrSortedColIndC,
                                         pBufferSizeInBytes)
    @check @runtime_ccall((:cusparseDcsrgeam2_bufferSizeExt, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, Ptr{Cdouble}, cusparseMatDescr_t, Cint,
                  CuPtr{Cdouble}, CuPtr{Cint}, CuPtr{Cint}, Ptr{Cdouble},
                  cusparseMatDescr_t, Cint, CuPtr{Cdouble}, CuPtr{Cint}, CuPtr{Cint},
                  cusparseMatDescr_t, CuPtr{Cdouble}, CuPtr{Cint}, CuPtr{Cint},
                  Ptr{Csize_t}),
                 handle, m, n, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA,
                 csrSortedColIndA, beta, descrB, nnzB, csrSortedValB, csrSortedRowPtrB,
                 csrSortedColIndB, descrC, csrSortedValC, csrSortedRowPtrC,
                 csrSortedColIndC, pBufferSizeInBytes)
end

function cusparseCcsrgeam2_bufferSizeExt(handle, m, n, alpha, descrA, nnzA, csrSortedValA,
                                         csrSortedRowPtrA, csrSortedColIndA, beta, descrB,
                                         nnzB, csrSortedValB, csrSortedRowPtrB,
                                         csrSortedColIndB, descrC, csrSortedValC,
                                         csrSortedRowPtrC, csrSortedColIndC,
                                         pBufferSizeInBytes)
    @check @runtime_ccall((:cusparseCcsrgeam2_bufferSizeExt, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, Ptr{cuComplex}, cusparseMatDescr_t, Cint,
                  CuPtr{cuComplex}, CuPtr{Cint}, CuPtr{Cint}, Ptr{cuComplex},
                  cusparseMatDescr_t, Cint, CuPtr{cuComplex}, CuPtr{Cint}, CuPtr{Cint},
                  cusparseMatDescr_t, CuPtr{cuComplex}, CuPtr{Cint}, CuPtr{Cint},
                  Ptr{Csize_t}),
                 handle, m, n, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA,
                 csrSortedColIndA, beta, descrB, nnzB, csrSortedValB, csrSortedRowPtrB,
                 csrSortedColIndB, descrC, csrSortedValC, csrSortedRowPtrC,
                 csrSortedColIndC, pBufferSizeInBytes)
end

function cusparseZcsrgeam2_bufferSizeExt(handle, m, n, alpha, descrA, nnzA, csrSortedValA,
                                         csrSortedRowPtrA, csrSortedColIndA, beta, descrB,
                                         nnzB, csrSortedValB, csrSortedRowPtrB,
                                         csrSortedColIndB, descrC, csrSortedValC,
                                         csrSortedRowPtrC, csrSortedColIndC,
                                         pBufferSizeInBytes)
    @check @runtime_ccall((:cusparseZcsrgeam2_bufferSizeExt, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, Ptr{cuDoubleComplex}, cusparseMatDescr_t,
                  Cint, CuPtr{cuDoubleComplex}, CuPtr{Cint}, CuPtr{Cint},
                  Ptr{cuDoubleComplex}, cusparseMatDescr_t, Cint, CuPtr{cuDoubleComplex},
                  CuPtr{Cint}, CuPtr{Cint}, cusparseMatDescr_t, CuPtr{cuDoubleComplex},
                  CuPtr{Cint}, CuPtr{Cint}, Ptr{Csize_t}),
                 handle, m, n, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA,
                 csrSortedColIndA, beta, descrB, nnzB, csrSortedValB, csrSortedRowPtrB,
                 csrSortedColIndB, descrC, csrSortedValC, csrSortedRowPtrC,
                 csrSortedColIndC, pBufferSizeInBytes)
end

function cusparseXcsrgeam2Nnz(handle, m, n, descrA, nnzA, csrSortedRowPtrA,
                              csrSortedColIndA, descrB, nnzB, csrSortedRowPtrB,
                              csrSortedColIndB, descrC, csrSortedRowPtrC,
                              nnzTotalDevHostPtr, workspace)
    @check @runtime_ccall((:cusparseXcsrgeam2Nnz, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, cusparseMatDescr_t, Cint, CuPtr{Cint},
                  CuPtr{Cint}, cusparseMatDescr_t, Cint, CuPtr{Cint}, CuPtr{Cint},
                  cusparseMatDescr_t, CuPtr{Cint}, PtrOrCuPtr{Cint}, CuPtr{Cvoid}),
                 handle, m, n, descrA, nnzA, csrSortedRowPtrA, csrSortedColIndA, descrB,
                 nnzB, csrSortedRowPtrB, csrSortedColIndB, descrC, csrSortedRowPtrC,
                 nnzTotalDevHostPtr, workspace)
end

function cusparseScsrgeam2(handle, m, n, alpha, descrA, nnzA, csrSortedValA,
                           csrSortedRowPtrA, csrSortedColIndA, beta, descrB, nnzB,
                           csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, descrC,
                           csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBuffer)
    @check @runtime_ccall((:cusparseScsrgeam2, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, Ptr{Cfloat}, cusparseMatDescr_t, Cint,
                  CuPtr{Cfloat}, CuPtr{Cint}, CuPtr{Cint}, Ptr{Cfloat}, cusparseMatDescr_t,
                  Cint, CuPtr{Cfloat}, CuPtr{Cint}, CuPtr{Cint}, cusparseMatDescr_t,
                  CuPtr{Cfloat}, CuPtr{Cint}, CuPtr{Cint}, CuPtr{Cvoid}),
                 handle, m, n, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA,
                 csrSortedColIndA, beta, descrB, nnzB, csrSortedValB, csrSortedRowPtrB,
                 csrSortedColIndB, descrC, csrSortedValC, csrSortedRowPtrC,
                 csrSortedColIndC, pBuffer)
end

function cusparseDcsrgeam2(handle, m, n, alpha, descrA, nnzA, csrSortedValA,
                           csrSortedRowPtrA, csrSortedColIndA, beta, descrB, nnzB,
                           csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, descrC,
                           csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBuffer)
    @check @runtime_ccall((:cusparseDcsrgeam2, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, Ptr{Cdouble}, cusparseMatDescr_t, Cint,
                  CuPtr{Cdouble}, CuPtr{Cint}, CuPtr{Cint}, Ptr{Cdouble},
                  cusparseMatDescr_t, Cint, CuPtr{Cdouble}, CuPtr{Cint}, CuPtr{Cint},
                  cusparseMatDescr_t, CuPtr{Cdouble}, CuPtr{Cint}, CuPtr{Cint},
                  CuPtr{Cvoid}),
                 handle, m, n, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA,
                 csrSortedColIndA, beta, descrB, nnzB, csrSortedValB, csrSortedRowPtrB,
                 csrSortedColIndB, descrC, csrSortedValC, csrSortedRowPtrC,
                 csrSortedColIndC, pBuffer)
end

function cusparseCcsrgeam2(handle, m, n, alpha, descrA, nnzA, csrSortedValA,
                           csrSortedRowPtrA, csrSortedColIndA, beta, descrB, nnzB,
                           csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, descrC,
                           csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBuffer)
    @check @runtime_ccall((:cusparseCcsrgeam2, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, Ptr{cuComplex}, cusparseMatDescr_t, Cint,
                  CuPtr{cuComplex}, CuPtr{Cint}, CuPtr{Cint}, Ptr{cuComplex},
                  cusparseMatDescr_t, Cint, CuPtr{cuComplex}, CuPtr{Cint}, CuPtr{Cint},
                  cusparseMatDescr_t, CuPtr{cuComplex}, CuPtr{Cint}, CuPtr{Cint},
                  CuPtr{Cvoid}),
                 handle, m, n, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA,
                 csrSortedColIndA, beta, descrB, nnzB, csrSortedValB, csrSortedRowPtrB,
                 csrSortedColIndB, descrC, csrSortedValC, csrSortedRowPtrC,
                 csrSortedColIndC, pBuffer)
end

function cusparseZcsrgeam2(handle, m, n, alpha, descrA, nnzA, csrSortedValA,
                           csrSortedRowPtrA, csrSortedColIndA, beta, descrB, nnzB,
                           csrSortedValB, csrSortedRowPtrB, csrSortedColIndB, descrC,
                           csrSortedValC, csrSortedRowPtrC, csrSortedColIndC, pBuffer)
    @check @runtime_ccall((:cusparseZcsrgeam2, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, Ptr{cuDoubleComplex}, cusparseMatDescr_t,
                  Cint, CuPtr{cuDoubleComplex}, CuPtr{Cint}, CuPtr{Cint},
                  Ptr{cuDoubleComplex}, cusparseMatDescr_t, Cint, CuPtr{cuDoubleComplex},
                  CuPtr{Cint}, CuPtr{Cint}, cusparseMatDescr_t, CuPtr{cuDoubleComplex},
                  CuPtr{Cint}, CuPtr{Cint}, CuPtr{Cvoid}),
                 handle, m, n, alpha, descrA, nnzA, csrSortedValA, csrSortedRowPtrA,
                 csrSortedColIndA, beta, descrB, nnzB, csrSortedValB, csrSortedRowPtrB,
                 csrSortedColIndB, descrC, csrSortedValC, csrSortedRowPtrC,
                 csrSortedColIndC, pBuffer)
end

function cusparseScsrcolor(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA,
                           csrSortedColIndA, fractionToColor, ncolors, coloring,
                           reordering, info)
    @check @runtime_ccall((:cusparseScsrcolor, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, cusparseMatDescr_t, CuPtr{Cfloat},
                  CuPtr{Cint}, CuPtr{Cint}, Ptr{Cfloat}, Ptr{Cint}, CuPtr{Cint},
                  CuPtr{Cint}, cusparseColorInfo_t),
                 handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA,
                 fractionToColor, ncolors, coloring, reordering, info)
end

function cusparseDcsrcolor(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA,
                           csrSortedColIndA, fractionToColor, ncolors, coloring,
                           reordering, info)
    @check @runtime_ccall((:cusparseDcsrcolor, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, cusparseMatDescr_t, CuPtr{Cdouble},
                  CuPtr{Cint}, CuPtr{Cint}, Ptr{Cdouble}, Ptr{Cint}, CuPtr{Cint},
                  CuPtr{Cint}, cusparseColorInfo_t),
                 handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA,
                 fractionToColor, ncolors, coloring, reordering, info)
end

function cusparseCcsrcolor(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA,
                           csrSortedColIndA, fractionToColor, ncolors, coloring,
                           reordering, info)
    @check @runtime_ccall((:cusparseCcsrcolor, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, cusparseMatDescr_t, CuPtr{cuComplex},
                  CuPtr{Cint}, CuPtr{Cint}, Ptr{Cfloat}, Ptr{Cint}, CuPtr{Cint},
                  CuPtr{Cint}, cusparseColorInfo_t),
                 handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA,
                 fractionToColor, ncolors, coloring, reordering, info)
end

function cusparseZcsrcolor(handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA,
                           csrSortedColIndA, fractionToColor, ncolors, coloring,
                           reordering, info)
    @check @runtime_ccall((:cusparseZcsrcolor, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{cuDoubleComplex}, CuPtr{Cint}, CuPtr{Cint}, Ptr{Cdouble},
                  Ptr{Cint}, CuPtr{Cint}, CuPtr{Cint}, cusparseColorInfo_t),
                 handle, m, nnz, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA,
                 fractionToColor, ncolors, coloring, reordering, info)
end

function cusparseSnnz(handle, dirA, m, n, descrA, A, lda, nnzPerRowCol, nnzTotalDevHostPtr)
    @check @runtime_ccall((:cusparseSnnz, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{Cfloat}, Cint, CuPtr{Cint}, PtrOrCuPtr{Cint}),
                 handle, dirA, m, n, descrA, A, lda, nnzPerRowCol, nnzTotalDevHostPtr)
end

function cusparseDnnz(handle, dirA, m, n, descrA, A, lda, nnzPerRowCol, nnzTotalDevHostPtr)
    @check @runtime_ccall((:cusparseDnnz, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{Cdouble}, Cint, CuPtr{Cint}, PtrOrCuPtr{Cint}),
                 handle, dirA, m, n, descrA, A, lda, nnzPerRowCol, nnzTotalDevHostPtr)
end

function cusparseCnnz(handle, dirA, m, n, descrA, A, lda, nnzPerRowCol, nnzTotalDevHostPtr)
    @check @runtime_ccall((:cusparseCnnz, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{cuComplex}, Cint, CuPtr{Cint}, PtrOrCuPtr{Cint}),
                 handle, dirA, m, n, descrA, A, lda, nnzPerRowCol, nnzTotalDevHostPtr)
end

function cusparseZnnz(handle, dirA, m, n, descrA, A, lda, nnzPerRowCol, nnzTotalDevHostPtr)
    @check @runtime_ccall((:cusparseZnnz, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{cuDoubleComplex}, Cint, CuPtr{Cint}, PtrOrCuPtr{Cint}),
                 handle, dirA, m, n, descrA, A, lda, nnzPerRowCol, nnzTotalDevHostPtr)
end

function cusparseSnnz_compress(handle, m, descr, csrSortedValA, csrSortedRowPtrA,
                               nnzPerRow, nnzC, tol)
    @check @runtime_ccall((:cusparseSnnz_compress, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, cusparseMatDescr_t, CuPtr{Cfloat}, CuPtr{Cint},
                  CuPtr{Cint}, PtrOrCuPtr{Cint}, Cfloat),
                 handle, m, descr, csrSortedValA, csrSortedRowPtrA, nnzPerRow, nnzC, tol)
end

function cusparseDnnz_compress(handle, m, descr, csrSortedValA, csrSortedRowPtrA,
                               nnzPerRow, nnzC, tol)
    @check @runtime_ccall((:cusparseDnnz_compress, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, cusparseMatDescr_t, CuPtr{Cdouble}, CuPtr{Cint},
                  CuPtr{Cint}, PtrOrCuPtr{Cint}, Cdouble),
                 handle, m, descr, csrSortedValA, csrSortedRowPtrA, nnzPerRow, nnzC, tol)
end

function cusparseCnnz_compress(handle, m, descr, csrSortedValA, csrSortedRowPtrA,
                               nnzPerRow, nnzC, tol)
    @check @runtime_ccall((:cusparseCnnz_compress, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, cusparseMatDescr_t, CuPtr{cuComplex},
                  CuPtr{Cint}, CuPtr{Cint}, PtrOrCuPtr{Cint}, cuComplex),
                 handle, m, descr, csrSortedValA, csrSortedRowPtrA, nnzPerRow, nnzC, tol)
end

function cusparseZnnz_compress(handle, m, descr, csrSortedValA, csrSortedRowPtrA,
                               nnzPerRow, nnzC, tol)
    @check @runtime_ccall((:cusparseZnnz_compress, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, cusparseMatDescr_t, CuPtr{cuDoubleComplex},
                  CuPtr{Cint}, CuPtr{Cint}, PtrOrCuPtr{Cint}, cuDoubleComplex),
                 handle, m, descr, csrSortedValA, csrSortedRowPtrA, nnzPerRow, nnzC, tol)
end

function cusparseScsr2csr_compress(handle, m, n, descrA, csrSortedValA, csrSortedColIndA,
                                   csrSortedRowPtrA, nnzA, nnzPerRow, csrSortedValC,
                                   csrSortedColIndC, csrSortedRowPtrC, tol)
    @check @runtime_ccall((:cusparseScsr2csr_compress, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, cusparseMatDescr_t, CuPtr{Cfloat},
                  CuPtr{Cint}, CuPtr{Cint}, Cint, CuPtr{Cint}, CuPtr{Cfloat}, CuPtr{Cint},
                  CuPtr{Cint}, Cfloat),
                 handle, m, n, descrA, csrSortedValA, csrSortedColIndA, csrSortedRowPtrA,
                 nnzA, nnzPerRow, csrSortedValC, csrSortedColIndC, csrSortedRowPtrC, tol)
end

function cusparseDcsr2csr_compress(handle, m, n, descrA, csrSortedValA, csrSortedColIndA,
                                   csrSortedRowPtrA, nnzA, nnzPerRow, csrSortedValC,
                                   csrSortedColIndC, csrSortedRowPtrC, tol)
    @check @runtime_ccall((:cusparseDcsr2csr_compress, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, cusparseMatDescr_t, CuPtr{Cdouble},
                  CuPtr{Cint}, CuPtr{Cint}, Cint, CuPtr{Cint}, CuPtr{Cdouble}, CuPtr{Cint},
                  CuPtr{Cint}, Cdouble),
                 handle, m, n, descrA, csrSortedValA, csrSortedColIndA, csrSortedRowPtrA,
                 nnzA, nnzPerRow, csrSortedValC, csrSortedColIndC, csrSortedRowPtrC, tol)
end

function cusparseCcsr2csr_compress(handle, m, n, descrA, csrSortedValA, csrSortedColIndA,
                                   csrSortedRowPtrA, nnzA, nnzPerRow, csrSortedValC,
                                   csrSortedColIndC, csrSortedRowPtrC, tol)
    @check @runtime_ccall((:cusparseCcsr2csr_compress, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, cusparseMatDescr_t, CuPtr{cuComplex},
                  CuPtr{Cint}, CuPtr{Cint}, Cint, CuPtr{Cint}, CuPtr{cuComplex},
                  CuPtr{Cint}, CuPtr{Cint}, cuComplex),
                 handle, m, n, descrA, csrSortedValA, csrSortedColIndA, csrSortedRowPtrA,
                 nnzA, nnzPerRow, csrSortedValC, csrSortedColIndC, csrSortedRowPtrC, tol)
end

function cusparseZcsr2csr_compress(handle, m, n, descrA, csrSortedValA, csrSortedColIndA,
                                   csrSortedRowPtrA, nnzA, nnzPerRow, csrSortedValC,
                                   csrSortedColIndC, csrSortedRowPtrC, tol)
    @check @runtime_ccall((:cusparseZcsr2csr_compress, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{cuDoubleComplex}, CuPtr{Cint}, CuPtr{Cint}, Cint, CuPtr{Cint},
                  CuPtr{cuDoubleComplex}, CuPtr{Cint}, CuPtr{Cint}, cuDoubleComplex),
                 handle, m, n, descrA, csrSortedValA, csrSortedColIndA, csrSortedRowPtrA,
                 nnzA, nnzPerRow, csrSortedValC, csrSortedColIndC, csrSortedRowPtrC, tol)
end

function cusparseSdense2csr(handle, m, n, descrA, A, lda, nnzPerRow, csrSortedValA,
                            csrSortedRowPtrA, csrSortedColIndA)
    @check @runtime_ccall((:cusparseSdense2csr, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, cusparseMatDescr_t, CuPtr{Cfloat}, Cint,
                  CuPtr{Cint}, CuPtr{Cfloat}, CuPtr{Cint}, CuPtr{Cint}),
                 handle, m, n, descrA, A, lda, nnzPerRow, csrSortedValA, csrSortedRowPtrA,
                 csrSortedColIndA)
end

function cusparseDdense2csr(handle, m, n, descrA, A, lda, nnzPerRow, csrSortedValA,
                            csrSortedRowPtrA, csrSortedColIndA)
    @check @runtime_ccall((:cusparseDdense2csr, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, cusparseMatDescr_t, CuPtr{Cdouble}, Cint,
                  CuPtr{Cint}, CuPtr{Cdouble}, CuPtr{Cint}, CuPtr{Cint}),
                 handle, m, n, descrA, A, lda, nnzPerRow, csrSortedValA, csrSortedRowPtrA,
                 csrSortedColIndA)
end

function cusparseCdense2csr(handle, m, n, descrA, A, lda, nnzPerRow, csrSortedValA,
                            csrSortedRowPtrA, csrSortedColIndA)
    @check @runtime_ccall((:cusparseCdense2csr, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, cusparseMatDescr_t, CuPtr{cuComplex},
                  Cint, CuPtr{Cint}, CuPtr{cuComplex}, CuPtr{Cint}, CuPtr{Cint}),
                 handle, m, n, descrA, A, lda, nnzPerRow, csrSortedValA, csrSortedRowPtrA,
                 csrSortedColIndA)
end

function cusparseZdense2csr(handle, m, n, descrA, A, lda, nnzPerRow, csrSortedValA,
                            csrSortedRowPtrA, csrSortedColIndA)
    @check @runtime_ccall((:cusparseZdense2csr, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{cuDoubleComplex}, Cint, CuPtr{Cint}, CuPtr{cuDoubleComplex},
                  CuPtr{Cint}, CuPtr{Cint}),
                 handle, m, n, descrA, A, lda, nnzPerRow, csrSortedValA, csrSortedRowPtrA,
                 csrSortedColIndA)
end

function cusparseScsr2dense(handle, m, n, descrA, csrSortedValA, csrSortedRowPtrA,
                            csrSortedColIndA, A, lda)
    @check @runtime_ccall((:cusparseScsr2dense, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, cusparseMatDescr_t, CuPtr{Cfloat},
                  CuPtr{Cint}, CuPtr{Cint}, CuPtr{Cfloat}, Cint),
                 handle, m, n, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA,
                 A, lda)
end

function cusparseDcsr2dense(handle, m, n, descrA, csrSortedValA, csrSortedRowPtrA,
                            csrSortedColIndA, A, lda)
    @check @runtime_ccall((:cusparseDcsr2dense, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, cusparseMatDescr_t, CuPtr{Cdouble},
                  CuPtr{Cint}, CuPtr{Cint}, CuPtr{Cdouble}, Cint),
                 handle, m, n, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA,
                 A, lda)
end

function cusparseCcsr2dense(handle, m, n, descrA, csrSortedValA, csrSortedRowPtrA,
                            csrSortedColIndA, A, lda)
    @check @runtime_ccall((:cusparseCcsr2dense, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, cusparseMatDescr_t, CuPtr{cuComplex},
                  CuPtr{Cint}, CuPtr{Cint}, CuPtr{cuComplex}, Cint),
                 handle, m, n, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA,
                 A, lda)
end

function cusparseZcsr2dense(handle, m, n, descrA, csrSortedValA, csrSortedRowPtrA,
                            csrSortedColIndA, A, lda)
    @check @runtime_ccall((:cusparseZcsr2dense, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{cuDoubleComplex}, CuPtr{Cint}, CuPtr{Cint}, CuPtr{cuDoubleComplex},
                  Cint),
                 handle, m, n, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA,
                 A, lda)
end

function cusparseSdense2csc(handle, m, n, descrA, A, lda, nnzPerCol, cscSortedValA,
                            cscSortedRowIndA, cscSortedColPtrA)
    @check @runtime_ccall((:cusparseSdense2csc, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, cusparseMatDescr_t, CuPtr{Cfloat}, Cint,
                  CuPtr{Cint}, CuPtr{Cfloat}, CuPtr{Cint}, CuPtr{Cint}),
                 handle, m, n, descrA, A, lda, nnzPerCol, cscSortedValA, cscSortedRowIndA,
                 cscSortedColPtrA)
end

function cusparseDdense2csc(handle, m, n, descrA, A, lda, nnzPerCol, cscSortedValA,
                            cscSortedRowIndA, cscSortedColPtrA)
    @check @runtime_ccall((:cusparseDdense2csc, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, cusparseMatDescr_t, CuPtr{Cdouble}, Cint,
                  CuPtr{Cint}, CuPtr{Cdouble}, CuPtr{Cint}, CuPtr{Cint}),
                 handle, m, n, descrA, A, lda, nnzPerCol, cscSortedValA, cscSortedRowIndA,
                 cscSortedColPtrA)
end

function cusparseCdense2csc(handle, m, n, descrA, A, lda, nnzPerCol, cscSortedValA,
                            cscSortedRowIndA, cscSortedColPtrA)
    @check @runtime_ccall((:cusparseCdense2csc, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, cusparseMatDescr_t, CuPtr{cuComplex},
                  Cint, CuPtr{Cint}, CuPtr{cuComplex}, CuPtr{Cint}, CuPtr{Cint}),
                 handle, m, n, descrA, A, lda, nnzPerCol, cscSortedValA, cscSortedRowIndA,
                 cscSortedColPtrA)
end

function cusparseZdense2csc(handle, m, n, descrA, A, lda, nnzPerCol, cscSortedValA,
                            cscSortedRowIndA, cscSortedColPtrA)
    @check @runtime_ccall((:cusparseZdense2csc, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{cuDoubleComplex}, Cint, CuPtr{Cint}, CuPtr{cuDoubleComplex},
                  CuPtr{Cint}, CuPtr{Cint}),
                 handle, m, n, descrA, A, lda, nnzPerCol, cscSortedValA, cscSortedRowIndA,
                 cscSortedColPtrA)
end

function cusparseScsc2dense(handle, m, n, descrA, cscSortedValA, cscSortedRowIndA,
                            cscSortedColPtrA, A, lda)
    @check @runtime_ccall((:cusparseScsc2dense, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, cusparseMatDescr_t, CuPtr{Cfloat},
                  CuPtr{Cint}, CuPtr{Cint}, CuPtr{Cfloat}, Cint),
                 handle, m, n, descrA, cscSortedValA, cscSortedRowIndA, cscSortedColPtrA,
                 A, lda)
end

function cusparseDcsc2dense(handle, m, n, descrA, cscSortedValA, cscSortedRowIndA,
                            cscSortedColPtrA, A, lda)
    @check @runtime_ccall((:cusparseDcsc2dense, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, cusparseMatDescr_t, CuPtr{Cdouble},
                  CuPtr{Cint}, CuPtr{Cint}, CuPtr{Cdouble}, Cint),
                 handle, m, n, descrA, cscSortedValA, cscSortedRowIndA, cscSortedColPtrA,
                 A, lda)
end

function cusparseCcsc2dense(handle, m, n, descrA, cscSortedValA, cscSortedRowIndA,
                            cscSortedColPtrA, A, lda)
    @check @runtime_ccall((:cusparseCcsc2dense, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, cusparseMatDescr_t, CuPtr{cuComplex},
                  CuPtr{Cint}, CuPtr{Cint}, CuPtr{cuComplex}, Cint),
                 handle, m, n, descrA, cscSortedValA, cscSortedRowIndA, cscSortedColPtrA,
                 A, lda)
end

function cusparseZcsc2dense(handle, m, n, descrA, cscSortedValA, cscSortedRowIndA,
                            cscSortedColPtrA, A, lda)
    @check @runtime_ccall((:cusparseZcsc2dense, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{cuDoubleComplex}, CuPtr{Cint}, CuPtr{Cint}, CuPtr{cuDoubleComplex},
                  Cint),
                 handle, m, n, descrA, cscSortedValA, cscSortedRowIndA, cscSortedColPtrA,
                 A, lda)
end

function cusparseXcoo2csr(handle, cooRowInd, nnz, m, csrSortedRowPtr, idxBase)
    @check @runtime_ccall((:cusparseXcoo2csr, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, CuPtr{Cint}, Cint, Cint, CuPtr{Cint},
                  cusparseIndexBase_t),
                 handle, cooRowInd, nnz, m, csrSortedRowPtr, idxBase)
end

function cusparseXcsr2coo(handle, csrSortedRowPtr, nnz, m, cooRowInd, idxBase)
    @check @runtime_ccall((:cusparseXcsr2coo, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, CuPtr{Cint}, Cint, Cint, CuPtr{Cint},
                  cusparseIndexBase_t),
                 handle, csrSortedRowPtr, nnz, m, cooRowInd, idxBase)
end

function cusparseCsr2cscEx(handle, m, n, nnz, csrSortedVal, csrSortedValtype,
                           csrSortedRowPtr, csrSortedColInd, cscSortedVal,
                           cscSortedValtype, cscSortedRowInd, cscSortedColPtr, copyValues,
                           idxBase, executiontype)
    @check @runtime_ccall((:cusparseCsr2cscEx, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, Cint, CuPtr{Cvoid}, cudaDataType,
                  CuPtr{Cint}, CuPtr{Cint}, CuPtr{Cvoid}, cudaDataType, CuPtr{Cint},
                  CuPtr{Cint}, cusparseAction_t, cusparseIndexBase_t, cudaDataType),
                 handle, m, n, nnz, csrSortedVal, csrSortedValtype, csrSortedRowPtr,
                 csrSortedColInd, cscSortedVal, cscSortedValtype, cscSortedRowInd,
                 cscSortedColPtr, copyValues, idxBase, executiontype)
end

function cusparseScsr2csc(handle, m, n, nnz, csrSortedVal, csrSortedRowPtr,
                          csrSortedColInd, cscSortedVal, cscSortedRowInd, cscSortedColPtr,
                          copyValues, idxBase)
    @check @runtime_ccall((:cusparseScsr2csc, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, Cint, CuPtr{Cfloat}, CuPtr{Cint},
                  CuPtr{Cint}, CuPtr{Cfloat}, CuPtr{Cint}, CuPtr{Cint}, cusparseAction_t,
                  cusparseIndexBase_t),
                 handle, m, n, nnz, csrSortedVal, csrSortedRowPtr, csrSortedColInd,
                 cscSortedVal, cscSortedRowInd, cscSortedColPtr, copyValues, idxBase)
end

function cusparseDcsr2csc(handle, m, n, nnz, csrSortedVal, csrSortedRowPtr,
                          csrSortedColInd, cscSortedVal, cscSortedRowInd, cscSortedColPtr,
                          copyValues, idxBase)
    @check @runtime_ccall((:cusparseDcsr2csc, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, Cint, CuPtr{Cdouble}, CuPtr{Cint},
                  CuPtr{Cint}, CuPtr{Cdouble}, CuPtr{Cint}, CuPtr{Cint}, cusparseAction_t,
                  cusparseIndexBase_t),
                 handle, m, n, nnz, csrSortedVal, csrSortedRowPtr, csrSortedColInd,
                 cscSortedVal, cscSortedRowInd, cscSortedColPtr, copyValues, idxBase)
end

function cusparseCcsr2csc(handle, m, n, nnz, csrSortedVal, csrSortedRowPtr,
                          csrSortedColInd, cscSortedVal, cscSortedRowInd, cscSortedColPtr,
                          copyValues, idxBase)
    @check @runtime_ccall((:cusparseCcsr2csc, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, Cint, CuPtr{cuComplex}, CuPtr{Cint},
                  CuPtr{Cint}, CuPtr{cuComplex}, CuPtr{Cint}, CuPtr{Cint},
                  cusparseAction_t, cusparseIndexBase_t),
                 handle, m, n, nnz, csrSortedVal, csrSortedRowPtr, csrSortedColInd,
                 cscSortedVal, cscSortedRowInd, cscSortedColPtr, copyValues, idxBase)
end

function cusparseZcsr2csc(handle, m, n, nnz, csrSortedVal, csrSortedRowPtr,
                          csrSortedColInd, cscSortedVal, cscSortedRowInd, cscSortedColPtr,
                          copyValues, idxBase)
    @check @runtime_ccall((:cusparseZcsr2csc, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, Cint, CuPtr{cuDoubleComplex}, CuPtr{Cint},
                  CuPtr{Cint}, CuPtr{cuDoubleComplex}, CuPtr{Cint}, CuPtr{Cint},
                  cusparseAction_t, cusparseIndexBase_t),
                 handle, m, n, nnz, csrSortedVal, csrSortedRowPtr, csrSortedColInd,
                 cscSortedVal, cscSortedRowInd, cscSortedColPtr, copyValues, idxBase)
end

function cusparseSdense2hyb(handle, m, n, descrA, A, lda, nnzPerRow, hybA, userEllWidth,
                            partitionType)
    @check @runtime_ccall((:cusparseSdense2hyb, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, cusparseMatDescr_t, CuPtr{Cfloat}, Cint,
                  CuPtr{Cint}, cusparseHybMat_t, Cint, cusparseHybPartition_t),
                 handle, m, n, descrA, A, lda, nnzPerRow, hybA, userEllWidth,
                 partitionType)
end

function cusparseDdense2hyb(handle, m, n, descrA, A, lda, nnzPerRow, hybA, userEllWidth,
                            partitionType)
    @check @runtime_ccall((:cusparseDdense2hyb, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, cusparseMatDescr_t, CuPtr{Cdouble}, Cint,
                  CuPtr{Cint}, cusparseHybMat_t, Cint, cusparseHybPartition_t),
                 handle, m, n, descrA, A, lda, nnzPerRow, hybA, userEllWidth,
                 partitionType)
end

function cusparseCdense2hyb(handle, m, n, descrA, A, lda, nnzPerRow, hybA, userEllWidth,
                            partitionType)
    @check @runtime_ccall((:cusparseCdense2hyb, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, cusparseMatDescr_t, CuPtr{cuComplex},
                  Cint, CuPtr{Cint}, cusparseHybMat_t, Cint, cusparseHybPartition_t),
                 handle, m, n, descrA, A, lda, nnzPerRow, hybA, userEllWidth,
                 partitionType)
end

function cusparseZdense2hyb(handle, m, n, descrA, A, lda, nnzPerRow, hybA, userEllWidth,
                            partitionType)
    @check @runtime_ccall((:cusparseZdense2hyb, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{cuDoubleComplex}, Cint, CuPtr{Cint}, cusparseHybMat_t, Cint,
                  cusparseHybPartition_t),
                 handle, m, n, descrA, A, lda, nnzPerRow, hybA, userEllWidth,
                 partitionType)
end

function cusparseShyb2dense(handle, descrA, hybA, A, lda)
    @check @runtime_ccall((:cusparseShyb2dense, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseMatDescr_t, cusparseHybMat_t, CuPtr{Cfloat},
                  Cint),
                 handle, descrA, hybA, A, lda)
end

function cusparseDhyb2dense(handle, descrA, hybA, A, lda)
    @check @runtime_ccall((:cusparseDhyb2dense, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseMatDescr_t, cusparseHybMat_t, CuPtr{Cdouble},
                  Cint),
                 handle, descrA, hybA, A, lda)
end

function cusparseChyb2dense(handle, descrA, hybA, A, lda)
    @check @runtime_ccall((:cusparseChyb2dense, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseMatDescr_t, cusparseHybMat_t,
                  CuPtr{cuComplex}, Cint),
                 handle, descrA, hybA, A, lda)
end

function cusparseZhyb2dense(handle, descrA, hybA, A, lda)
    @check @runtime_ccall((:cusparseZhyb2dense, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseMatDescr_t, cusparseHybMat_t,
                  CuPtr{cuDoubleComplex}, Cint),
                 handle, descrA, hybA, A, lda)
end

function cusparseScsr2hyb(handle, m, n, descrA, csrSortedValA, csrSortedRowPtrA,
                          csrSortedColIndA, hybA, userEllWidth, partitionType)
    @check @runtime_ccall((:cusparseScsr2hyb, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, cusparseMatDescr_t, CuPtr{Cfloat},
                  CuPtr{Cint}, CuPtr{Cint}, cusparseHybMat_t, Cint, cusparseHybPartition_t),
                 handle, m, n, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA,
                 hybA, userEllWidth, partitionType)
end

function cusparseDcsr2hyb(handle, m, n, descrA, csrSortedValA, csrSortedRowPtrA,
                          csrSortedColIndA, hybA, userEllWidth, partitionType)
    @check @runtime_ccall((:cusparseDcsr2hyb, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, cusparseMatDescr_t, CuPtr{Cdouble},
                  CuPtr{Cint}, CuPtr{Cint}, cusparseHybMat_t, Cint, cusparseHybPartition_t),
                 handle, m, n, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA,
                 hybA, userEllWidth, partitionType)
end

function cusparseCcsr2hyb(handle, m, n, descrA, csrSortedValA, csrSortedRowPtrA,
                          csrSortedColIndA, hybA, userEllWidth, partitionType)
    @check @runtime_ccall((:cusparseCcsr2hyb, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, cusparseMatDescr_t, CuPtr{cuComplex},
                  CuPtr{Cint}, CuPtr{Cint}, cusparseHybMat_t, Cint, cusparseHybPartition_t),
                 handle, m, n, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA,
                 hybA, userEllWidth, partitionType)
end

function cusparseZcsr2hyb(handle, m, n, descrA, csrSortedValA, csrSortedRowPtrA,
                          csrSortedColIndA, hybA, userEllWidth, partitionType)
    @check @runtime_ccall((:cusparseZcsr2hyb, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{cuDoubleComplex}, CuPtr{Cint}, CuPtr{Cint}, cusparseHybMat_t, Cint,
                  cusparseHybPartition_t),
                 handle, m, n, descrA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA,
                 hybA, userEllWidth, partitionType)
end

function cusparseShyb2csr(handle, descrA, hybA, csrSortedValA, csrSortedRowPtrA,
                          csrSortedColIndA)
    @check @runtime_ccall((:cusparseShyb2csr, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseMatDescr_t, cusparseHybMat_t, CuPtr{Cfloat},
                  CuPtr{Cint}, CuPtr{Cint}),
                 handle, descrA, hybA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA)
end

function cusparseDhyb2csr(handle, descrA, hybA, csrSortedValA, csrSortedRowPtrA,
                          csrSortedColIndA)
    @check @runtime_ccall((:cusparseDhyb2csr, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseMatDescr_t, cusparseHybMat_t, CuPtr{Cdouble},
                  CuPtr{Cint}, CuPtr{Cint}),
                 handle, descrA, hybA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA)
end

function cusparseChyb2csr(handle, descrA, hybA, csrSortedValA, csrSortedRowPtrA,
                          csrSortedColIndA)
    @check @runtime_ccall((:cusparseChyb2csr, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseMatDescr_t, cusparseHybMat_t,
                  CuPtr{cuComplex}, CuPtr{Cint}, CuPtr{Cint}),
                 handle, descrA, hybA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA)
end

function cusparseZhyb2csr(handle, descrA, hybA, csrSortedValA, csrSortedRowPtrA,
                          csrSortedColIndA)
    @check @runtime_ccall((:cusparseZhyb2csr, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseMatDescr_t, cusparseHybMat_t,
                  CuPtr{cuDoubleComplex}, CuPtr{Cint}, CuPtr{Cint}),
                 handle, descrA, hybA, csrSortedValA, csrSortedRowPtrA, csrSortedColIndA)
end

function cusparseScsc2hyb(handle, m, n, descrA, cscSortedValA, cscSortedRowIndA,
                          cscSortedColPtrA, hybA, userEllWidth, partitionType)
    @check @runtime_ccall((:cusparseScsc2hyb, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, cusparseMatDescr_t, CuPtr{Cfloat},
                  CuPtr{Cint}, CuPtr{Cint}, cusparseHybMat_t, Cint, cusparseHybPartition_t),
                 handle, m, n, descrA, cscSortedValA, cscSortedRowIndA, cscSortedColPtrA,
                 hybA, userEllWidth, partitionType)
end

function cusparseDcsc2hyb(handle, m, n, descrA, cscSortedValA, cscSortedRowIndA,
                          cscSortedColPtrA, hybA, userEllWidth, partitionType)
    @check @runtime_ccall((:cusparseDcsc2hyb, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, cusparseMatDescr_t, CuPtr{Cdouble},
                  CuPtr{Cint}, CuPtr{Cint}, cusparseHybMat_t, Cint, cusparseHybPartition_t),
                 handle, m, n, descrA, cscSortedValA, cscSortedRowIndA, cscSortedColPtrA,
                 hybA, userEllWidth, partitionType)
end

function cusparseCcsc2hyb(handle, m, n, descrA, cscSortedValA, cscSortedRowIndA,
                          cscSortedColPtrA, hybA, userEllWidth, partitionType)
    @check @runtime_ccall((:cusparseCcsc2hyb, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, cusparseMatDescr_t, CuPtr{cuComplex},
                  CuPtr{Cint}, CuPtr{Cint}, cusparseHybMat_t, Cint, cusparseHybPartition_t),
                 handle, m, n, descrA, cscSortedValA, cscSortedRowIndA, cscSortedColPtrA,
                 hybA, userEllWidth, partitionType)
end

function cusparseZcsc2hyb(handle, m, n, descrA, cscSortedValA, cscSortedRowIndA,
                          cscSortedColPtrA, hybA, userEllWidth, partitionType)
    @check @runtime_ccall((:cusparseZcsc2hyb, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{cuDoubleComplex}, CuPtr{Cint}, CuPtr{Cint}, cusparseHybMat_t, Cint,
                  cusparseHybPartition_t),
                 handle, m, n, descrA, cscSortedValA, cscSortedRowIndA, cscSortedColPtrA,
                 hybA, userEllWidth, partitionType)
end

function cusparseShyb2csc(handle, descrA, hybA, cscSortedVal, cscSortedRowInd,
                          cscSortedColPtr)
    @check @runtime_ccall((:cusparseShyb2csc, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseMatDescr_t, cusparseHybMat_t, CuPtr{Cfloat},
                  CuPtr{Cint}, CuPtr{Cint}),
                 handle, descrA, hybA, cscSortedVal, cscSortedRowInd, cscSortedColPtr)
end

function cusparseDhyb2csc(handle, descrA, hybA, cscSortedVal, cscSortedRowInd,
                          cscSortedColPtr)
    @check @runtime_ccall((:cusparseDhyb2csc, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseMatDescr_t, cusparseHybMat_t, CuPtr{Cdouble},
                  CuPtr{Cint}, CuPtr{Cint}),
                 handle, descrA, hybA, cscSortedVal, cscSortedRowInd, cscSortedColPtr)
end

function cusparseChyb2csc(handle, descrA, hybA, cscSortedVal, cscSortedRowInd,
                          cscSortedColPtr)
    @check @runtime_ccall((:cusparseChyb2csc, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseMatDescr_t, cusparseHybMat_t,
                  CuPtr{cuComplex}, CuPtr{Cint}, CuPtr{Cint}),
                 handle, descrA, hybA, cscSortedVal, cscSortedRowInd, cscSortedColPtr)
end

function cusparseZhyb2csc(handle, descrA, hybA, cscSortedVal, cscSortedRowInd,
                          cscSortedColPtr)
    @check @runtime_ccall((:cusparseZhyb2csc, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseMatDescr_t, cusparseHybMat_t,
                  CuPtr{cuDoubleComplex}, CuPtr{Cint}, CuPtr{Cint}),
                 handle, descrA, hybA, cscSortedVal, cscSortedRowInd, cscSortedColPtr)
end

function cusparseXcsr2bsrNnz(handle, dirA, m, n, descrA, csrSortedRowPtrA,
                             csrSortedColIndA, blockDim, descrC, bsrSortedRowPtrC,
                             nnzTotalDevHostPtr)
    @check @runtime_ccall((:cusparseXcsr2bsrNnz, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{Cint}, CuPtr{Cint}, Cint, cusparseMatDescr_t, CuPtr{Cint},
                  PtrOrCuPtr{Cint}),
                 handle, dirA, m, n, descrA, csrSortedRowPtrA, csrSortedColIndA, blockDim,
                 descrC, bsrSortedRowPtrC, nnzTotalDevHostPtr)
end

function cusparseScsr2bsr(handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA,
                          csrSortedColIndA, blockDim, descrC, bsrSortedValC,
                          bsrSortedRowPtrC, bsrSortedColIndC)
    @check @runtime_ccall((:cusparseScsr2bsr, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{Cfloat}, CuPtr{Cint}, CuPtr{Cint}, Cint, cusparseMatDescr_t,
                  CuPtr{Cfloat}, CuPtr{Cint}, CuPtr{Cint}),
                 handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA,
                 csrSortedColIndA, blockDim, descrC, bsrSortedValC, bsrSortedRowPtrC,
                 bsrSortedColIndC)
end

function cusparseDcsr2bsr(handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA,
                          csrSortedColIndA, blockDim, descrC, bsrSortedValC,
                          bsrSortedRowPtrC, bsrSortedColIndC)
    @check @runtime_ccall((:cusparseDcsr2bsr, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{Cdouble}, CuPtr{Cint}, CuPtr{Cint}, Cint, cusparseMatDescr_t,
                  CuPtr{Cdouble}, CuPtr{Cint}, CuPtr{Cint}),
                 handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA,
                 csrSortedColIndA, blockDim, descrC, bsrSortedValC, bsrSortedRowPtrC,
                 bsrSortedColIndC)
end

function cusparseCcsr2bsr(handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA,
                          csrSortedColIndA, blockDim, descrC, bsrSortedValC,
                          bsrSortedRowPtrC, bsrSortedColIndC)
    @check @runtime_ccall((:cusparseCcsr2bsr, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{cuComplex}, CuPtr{Cint}, CuPtr{Cint}, Cint, cusparseMatDescr_t,
                  CuPtr{cuComplex}, CuPtr{Cint}, CuPtr{Cint}),
                 handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA,
                 csrSortedColIndA, blockDim, descrC, bsrSortedValC, bsrSortedRowPtrC,
                 bsrSortedColIndC)
end

function cusparseZcsr2bsr(handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA,
                          csrSortedColIndA, blockDim, descrC, bsrSortedValC,
                          bsrSortedRowPtrC, bsrSortedColIndC)
    @check @runtime_ccall((:cusparseZcsr2bsr, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{cuDoubleComplex}, CuPtr{Cint}, CuPtr{Cint}, Cint,
                  cusparseMatDescr_t, CuPtr{cuDoubleComplex}, CuPtr{Cint}, CuPtr{Cint}),
                 handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA,
                 csrSortedColIndA, blockDim, descrC, bsrSortedValC, bsrSortedRowPtrC,
                 bsrSortedColIndC)
end

function cusparseSbsr2csr(handle, dirA, mb, nb, descrA, bsrSortedValA, bsrSortedRowPtrA,
                          bsrSortedColIndA, blockDim, descrC, csrSortedValC,
                          csrSortedRowPtrC, csrSortedColIndC)
    @check @runtime_ccall((:cusparseSbsr2csr, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{Cfloat}, CuPtr{Cint}, CuPtr{Cint}, Cint, cusparseMatDescr_t,
                  CuPtr{Cfloat}, CuPtr{Cint}, CuPtr{Cint}),
                 handle, dirA, mb, nb, descrA, bsrSortedValA, bsrSortedRowPtrA,
                 bsrSortedColIndA, blockDim, descrC, csrSortedValC, csrSortedRowPtrC,
                 csrSortedColIndC)
end

function cusparseDbsr2csr(handle, dirA, mb, nb, descrA, bsrSortedValA, bsrSortedRowPtrA,
                          bsrSortedColIndA, blockDim, descrC, csrSortedValC,
                          csrSortedRowPtrC, csrSortedColIndC)
    @check @runtime_ccall((:cusparseDbsr2csr, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{Cdouble}, CuPtr{Cint}, CuPtr{Cint}, Cint, cusparseMatDescr_t,
                  CuPtr{Cdouble}, CuPtr{Cint}, CuPtr{Cint}),
                 handle, dirA, mb, nb, descrA, bsrSortedValA, bsrSortedRowPtrA,
                 bsrSortedColIndA, blockDim, descrC, csrSortedValC, csrSortedRowPtrC,
                 csrSortedColIndC)
end

function cusparseCbsr2csr(handle, dirA, mb, nb, descrA, bsrSortedValA, bsrSortedRowPtrA,
                          bsrSortedColIndA, blockDim, descrC, csrSortedValC,
                          csrSortedRowPtrC, csrSortedColIndC)
    @check @runtime_ccall((:cusparseCbsr2csr, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{cuComplex}, CuPtr{Cint}, CuPtr{Cint}, Cint, cusparseMatDescr_t,
                  CuPtr{cuComplex}, CuPtr{Cint}, CuPtr{Cint}),
                 handle, dirA, mb, nb, descrA, bsrSortedValA, bsrSortedRowPtrA,
                 bsrSortedColIndA, blockDim, descrC, csrSortedValC, csrSortedRowPtrC,
                 csrSortedColIndC)
end

function cusparseZbsr2csr(handle, dirA, mb, nb, descrA, bsrSortedValA, bsrSortedRowPtrA,
                          bsrSortedColIndA, blockDim, descrC, csrSortedValC,
                          csrSortedRowPtrC, csrSortedColIndC)
    @check @runtime_ccall((:cusparseZbsr2csr, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{cuDoubleComplex}, CuPtr{Cint}, CuPtr{Cint}, Cint,
                  cusparseMatDescr_t, CuPtr{cuDoubleComplex}, CuPtr{Cint}, CuPtr{Cint}),
                 handle, dirA, mb, nb, descrA, bsrSortedValA, bsrSortedRowPtrA,
                 bsrSortedColIndA, blockDim, descrC, csrSortedValC, csrSortedRowPtrC,
                 csrSortedColIndC)
end

function cusparseSgebsr2gebsc_bufferSize(handle, mb, nb, nnzb, bsrSortedVal,
                                         bsrSortedRowPtr, bsrSortedColInd, rowBlockDim,
                                         colBlockDim, pBufferSizeInBytes)
    @check @runtime_ccall((:cusparseSgebsr2gebsc_bufferSize, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, Cint, CuPtr{Cfloat}, CuPtr{Cint},
                  CuPtr{Cint}, Cint, Cint, Ptr{Cint}),
                 handle, mb, nb, nnzb, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd,
                 rowBlockDim, colBlockDim, pBufferSizeInBytes)
end

function cusparseDgebsr2gebsc_bufferSize(handle, mb, nb, nnzb, bsrSortedVal,
                                         bsrSortedRowPtr, bsrSortedColInd, rowBlockDim,
                                         colBlockDim, pBufferSizeInBytes)
    @check @runtime_ccall((:cusparseDgebsr2gebsc_bufferSize, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, Cint, CuPtr{Cdouble}, CuPtr{Cint},
                  CuPtr{Cint}, Cint, Cint, Ptr{Cint}),
                 handle, mb, nb, nnzb, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd,
                 rowBlockDim, colBlockDim, pBufferSizeInBytes)
end

function cusparseCgebsr2gebsc_bufferSize(handle, mb, nb, nnzb, bsrSortedVal,
                                         bsrSortedRowPtr, bsrSortedColInd, rowBlockDim,
                                         colBlockDim, pBufferSizeInBytes)
    @check @runtime_ccall((:cusparseCgebsr2gebsc_bufferSize, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, Cint, CuPtr{cuComplex}, CuPtr{Cint},
                  CuPtr{Cint}, Cint, Cint, Ptr{Cint}),
                 handle, mb, nb, nnzb, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd,
                 rowBlockDim, colBlockDim, pBufferSizeInBytes)
end

function cusparseZgebsr2gebsc_bufferSize(handle, mb, nb, nnzb, bsrSortedVal,
                                         bsrSortedRowPtr, bsrSortedColInd, rowBlockDim,
                                         colBlockDim, pBufferSizeInBytes)
    @check @runtime_ccall((:cusparseZgebsr2gebsc_bufferSize, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, Cint, CuPtr{cuDoubleComplex}, CuPtr{Cint},
                  CuPtr{Cint}, Cint, Cint, Ptr{Cint}),
                 handle, mb, nb, nnzb, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd,
                 rowBlockDim, colBlockDim, pBufferSizeInBytes)
end

function cusparseSgebsr2gebsc_bufferSizeExt(handle, mb, nb, nnzb, bsrSortedVal,
                                            bsrSortedRowPtr, bsrSortedColInd, rowBlockDim,
                                            colBlockDim, pBufferSize)
    @check @runtime_ccall((:cusparseSgebsr2gebsc_bufferSizeExt, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, Cint, CuPtr{Cfloat}, CuPtr{Cint},
                  CuPtr{Cint}, Cint, Cint, Ptr{Csize_t}),
                 handle, mb, nb, nnzb, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd,
                 rowBlockDim, colBlockDim, pBufferSize)
end

function cusparseDgebsr2gebsc_bufferSizeExt(handle, mb, nb, nnzb, bsrSortedVal,
                                            bsrSortedRowPtr, bsrSortedColInd, rowBlockDim,
                                            colBlockDim, pBufferSize)
    @check @runtime_ccall((:cusparseDgebsr2gebsc_bufferSizeExt, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, Cint, CuPtr{Cdouble}, CuPtr{Cint},
                  CuPtr{Cint}, Cint, Cint, Ptr{Csize_t}),
                 handle, mb, nb, nnzb, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd,
                 rowBlockDim, colBlockDim, pBufferSize)
end

function cusparseCgebsr2gebsc_bufferSizeExt(handle, mb, nb, nnzb, bsrSortedVal,
                                            bsrSortedRowPtr, bsrSortedColInd, rowBlockDim,
                                            colBlockDim, pBufferSize)
    @check @runtime_ccall((:cusparseCgebsr2gebsc_bufferSizeExt, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, Cint, CuPtr{cuComplex}, CuPtr{Cint},
                  CuPtr{Cint}, Cint, Cint, Ptr{Csize_t}),
                 handle, mb, nb, nnzb, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd,
                 rowBlockDim, colBlockDim, pBufferSize)
end

function cusparseZgebsr2gebsc_bufferSizeExt(handle, mb, nb, nnzb, bsrSortedVal,
                                            bsrSortedRowPtr, bsrSortedColInd, rowBlockDim,
                                            colBlockDim, pBufferSize)
    @check @runtime_ccall((:cusparseZgebsr2gebsc_bufferSizeExt, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, Cint, CuPtr{cuDoubleComplex}, CuPtr{Cint},
                  CuPtr{Cint}, Cint, Cint, Ptr{Csize_t}),
                 handle, mb, nb, nnzb, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd,
                 rowBlockDim, colBlockDim, pBufferSize)
end

function cusparseSgebsr2gebsc(handle, mb, nb, nnzb, bsrSortedVal, bsrSortedRowPtr,
                              bsrSortedColInd, rowBlockDim, colBlockDim, bscVal, bscRowInd,
                              bscColPtr, copyValues, idxBase, pBuffer)
    @check @runtime_ccall((:cusparseSgebsr2gebsc, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, Cint, CuPtr{Cfloat}, CuPtr{Cint},
                  CuPtr{Cint}, Cint, Cint, CuPtr{Cfloat}, CuPtr{Cint}, CuPtr{Cint},
                  cusparseAction_t, cusparseIndexBase_t, CuPtr{Cvoid}),
                 handle, mb, nb, nnzb, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd,
                 rowBlockDim, colBlockDim, bscVal, bscRowInd, bscColPtr, copyValues,
                 idxBase, pBuffer)
end

function cusparseDgebsr2gebsc(handle, mb, nb, nnzb, bsrSortedVal, bsrSortedRowPtr,
                              bsrSortedColInd, rowBlockDim, colBlockDim, bscVal, bscRowInd,
                              bscColPtr, copyValues, idxBase, pBuffer)
    @check @runtime_ccall((:cusparseDgebsr2gebsc, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, Cint, CuPtr{Cdouble}, CuPtr{Cint},
                  CuPtr{Cint}, Cint, Cint, CuPtr{Cdouble}, CuPtr{Cint}, CuPtr{Cint},
                  cusparseAction_t, cusparseIndexBase_t, CuPtr{Cvoid}),
                 handle, mb, nb, nnzb, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd,
                 rowBlockDim, colBlockDim, bscVal, bscRowInd, bscColPtr, copyValues,
                 idxBase, pBuffer)
end

function cusparseCgebsr2gebsc(handle, mb, nb, nnzb, bsrSortedVal, bsrSortedRowPtr,
                              bsrSortedColInd, rowBlockDim, colBlockDim, bscVal, bscRowInd,
                              bscColPtr, copyValues, idxBase, pBuffer)
    @check @runtime_ccall((:cusparseCgebsr2gebsc, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, Cint, CuPtr{cuComplex}, CuPtr{Cint},
                  CuPtr{Cint}, Cint, Cint, CuPtr{cuComplex}, CuPtr{Cint}, CuPtr{Cint},
                  cusparseAction_t, cusparseIndexBase_t, CuPtr{Cvoid}),
                 handle, mb, nb, nnzb, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd,
                 rowBlockDim, colBlockDim, bscVal, bscRowInd, bscColPtr, copyValues,
                 idxBase, pBuffer)
end

function cusparseZgebsr2gebsc(handle, mb, nb, nnzb, bsrSortedVal, bsrSortedRowPtr,
                              bsrSortedColInd, rowBlockDim, colBlockDim, bscVal, bscRowInd,
                              bscColPtr, copyValues, idxBase, pBuffer)
    @check @runtime_ccall((:cusparseZgebsr2gebsc, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, Cint, CuPtr{cuDoubleComplex}, CuPtr{Cint},
                  CuPtr{Cint}, Cint, Cint, CuPtr{cuDoubleComplex}, CuPtr{Cint},
                  CuPtr{Cint}, cusparseAction_t, cusparseIndexBase_t, CuPtr{Cvoid}),
                 handle, mb, nb, nnzb, bsrSortedVal, bsrSortedRowPtr, bsrSortedColInd,
                 rowBlockDim, colBlockDim, bscVal, bscRowInd, bscColPtr, copyValues,
                 idxBase, pBuffer)
end

function cusparseXgebsr2csr(handle, dirA, mb, nb, descrA, bsrSortedRowPtrA,
                            bsrSortedColIndA, rowBlockDim, colBlockDim, descrC,
                            csrSortedRowPtrC, csrSortedColIndC)
    @check @runtime_ccall((:cusparseXgebsr2csr, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{Cint}, CuPtr{Cint}, Cint, Cint, cusparseMatDescr_t, CuPtr{Cint},
                  CuPtr{Cint}),
                 handle, dirA, mb, nb, descrA, bsrSortedRowPtrA, bsrSortedColIndA,
                 rowBlockDim, colBlockDim, descrC, csrSortedRowPtrC, csrSortedColIndC)
end

function cusparseSgebsr2csr(handle, dirA, mb, nb, descrA, bsrSortedValA, bsrSortedRowPtrA,
                            bsrSortedColIndA, rowBlockDim, colBlockDim, descrC,
                            csrSortedValC, csrSortedRowPtrC, csrSortedColIndC)
    @check @runtime_ccall((:cusparseSgebsr2csr, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{Cfloat}, CuPtr{Cint}, CuPtr{Cint}, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{Cfloat}, CuPtr{Cint}, CuPtr{Cint}),
                 handle, dirA, mb, nb, descrA, bsrSortedValA, bsrSortedRowPtrA,
                 bsrSortedColIndA, rowBlockDim, colBlockDim, descrC, csrSortedValC,
                 csrSortedRowPtrC, csrSortedColIndC)
end

function cusparseDgebsr2csr(handle, dirA, mb, nb, descrA, bsrSortedValA, bsrSortedRowPtrA,
                            bsrSortedColIndA, rowBlockDim, colBlockDim, descrC,
                            csrSortedValC, csrSortedRowPtrC, csrSortedColIndC)
    @check @runtime_ccall((:cusparseDgebsr2csr, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{Cdouble}, CuPtr{Cint}, CuPtr{Cint}, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{Cdouble}, CuPtr{Cint}, CuPtr{Cint}),
                 handle, dirA, mb, nb, descrA, bsrSortedValA, bsrSortedRowPtrA,
                 bsrSortedColIndA, rowBlockDim, colBlockDim, descrC, csrSortedValC,
                 csrSortedRowPtrC, csrSortedColIndC)
end

function cusparseCgebsr2csr(handle, dirA, mb, nb, descrA, bsrSortedValA, bsrSortedRowPtrA,
                            bsrSortedColIndA, rowBlockDim, colBlockDim, descrC,
                            csrSortedValC, csrSortedRowPtrC, csrSortedColIndC)
    @check @runtime_ccall((:cusparseCgebsr2csr, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{cuComplex}, CuPtr{Cint}, CuPtr{Cint}, Cint, Cint,
                  cusparseMatDescr_t, CuPtr{cuComplex}, CuPtr{Cint}, CuPtr{Cint}),
                 handle, dirA, mb, nb, descrA, bsrSortedValA, bsrSortedRowPtrA,
                 bsrSortedColIndA, rowBlockDim, colBlockDim, descrC, csrSortedValC,
                 csrSortedRowPtrC, csrSortedColIndC)
end

function cusparseZgebsr2csr(handle, dirA, mb, nb, descrA, bsrSortedValA, bsrSortedRowPtrA,
                            bsrSortedColIndA, rowBlockDim, colBlockDim, descrC,
                            csrSortedValC, csrSortedRowPtrC, csrSortedColIndC)
    @check @runtime_ccall((:cusparseZgebsr2csr, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{cuDoubleComplex}, CuPtr{Cint}, CuPtr{Cint}, Cint, Cint,
                  cusparseMatDescr_t, CuPtr{cuDoubleComplex}, CuPtr{Cint}, CuPtr{Cint}),
                 handle, dirA, mb, nb, descrA, bsrSortedValA, bsrSortedRowPtrA,
                 bsrSortedColIndA, rowBlockDim, colBlockDim, descrC, csrSortedValC,
                 csrSortedRowPtrC, csrSortedColIndC)
end

function cusparseScsr2gebsr_bufferSize(handle, dirA, m, n, descrA, csrSortedValA,
                                       csrSortedRowPtrA, csrSortedColIndA, rowBlockDim,
                                       colBlockDim, pBufferSizeInBytes)
    @check @runtime_ccall((:cusparseScsr2gebsr_bufferSize, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{Cfloat}, CuPtr{Cint}, CuPtr{Cint}, Cint, Cint, Ptr{Cint}),
                 handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA,
                 csrSortedColIndA, rowBlockDim, colBlockDim, pBufferSizeInBytes)
end

function cusparseDcsr2gebsr_bufferSize(handle, dirA, m, n, descrA, csrSortedValA,
                                       csrSortedRowPtrA, csrSortedColIndA, rowBlockDim,
                                       colBlockDim, pBufferSizeInBytes)
    @check @runtime_ccall((:cusparseDcsr2gebsr_bufferSize, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{Cdouble}, CuPtr{Cint}, CuPtr{Cint}, Cint, Cint, Ptr{Cint}),
                 handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA,
                 csrSortedColIndA, rowBlockDim, colBlockDim, pBufferSizeInBytes)
end

function cusparseCcsr2gebsr_bufferSize(handle, dirA, m, n, descrA, csrSortedValA,
                                       csrSortedRowPtrA, csrSortedColIndA, rowBlockDim,
                                       colBlockDim, pBufferSizeInBytes)
    @check @runtime_ccall((:cusparseCcsr2gebsr_bufferSize, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{cuComplex}, CuPtr{Cint}, CuPtr{Cint}, Cint, Cint, Ptr{Cint}),
                 handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA,
                 csrSortedColIndA, rowBlockDim, colBlockDim, pBufferSizeInBytes)
end

function cusparseZcsr2gebsr_bufferSize(handle, dirA, m, n, descrA, csrSortedValA,
                                       csrSortedRowPtrA, csrSortedColIndA, rowBlockDim,
                                       colBlockDim, pBufferSizeInBytes)
    @check @runtime_ccall((:cusparseZcsr2gebsr_bufferSize, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{cuDoubleComplex}, CuPtr{Cint}, CuPtr{Cint}, Cint, Cint, Ptr{Cint}),
                 handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA,
                 csrSortedColIndA, rowBlockDim, colBlockDim, pBufferSizeInBytes)
end

function cusparseScsr2gebsr_bufferSizeExt(handle, dirA, m, n, descrA, csrSortedValA,
                                          csrSortedRowPtrA, csrSortedColIndA, rowBlockDim,
                                          colBlockDim, pBufferSize)
    @check @runtime_ccall((:cusparseScsr2gebsr_bufferSizeExt, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{Cfloat}, CuPtr{Cint}, CuPtr{Cint}, Cint, Cint, Ptr{Csize_t}),
                 handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA,
                 csrSortedColIndA, rowBlockDim, colBlockDim, pBufferSize)
end

function cusparseDcsr2gebsr_bufferSizeExt(handle, dirA, m, n, descrA, csrSortedValA,
                                          csrSortedRowPtrA, csrSortedColIndA, rowBlockDim,
                                          colBlockDim, pBufferSize)
    @check @runtime_ccall((:cusparseDcsr2gebsr_bufferSizeExt, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{Cdouble}, CuPtr{Cint}, CuPtr{Cint}, Cint, Cint, Ptr{Csize_t}),
                 handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA,
                 csrSortedColIndA, rowBlockDim, colBlockDim, pBufferSize)
end

function cusparseCcsr2gebsr_bufferSizeExt(handle, dirA, m, n, descrA, csrSortedValA,
                                          csrSortedRowPtrA, csrSortedColIndA, rowBlockDim,
                                          colBlockDim, pBufferSize)
    @check @runtime_ccall((:cusparseCcsr2gebsr_bufferSizeExt, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{cuComplex}, CuPtr{Cint}, CuPtr{Cint}, Cint, Cint, Ptr{Csize_t}),
                 handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA,
                 csrSortedColIndA, rowBlockDim, colBlockDim, pBufferSize)
end

function cusparseZcsr2gebsr_bufferSizeExt(handle, dirA, m, n, descrA, csrSortedValA,
                                          csrSortedRowPtrA, csrSortedColIndA, rowBlockDim,
                                          colBlockDim, pBufferSize)
    @check @runtime_ccall((:cusparseZcsr2gebsr_bufferSizeExt, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{cuDoubleComplex}, CuPtr{Cint}, CuPtr{Cint}, Cint, Cint,
                  Ptr{Csize_t}),
                 handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA,
                 csrSortedColIndA, rowBlockDim, colBlockDim, pBufferSize)
end

function cusparseXcsr2gebsrNnz(handle, dirA, m, n, descrA, csrSortedRowPtrA,
                               csrSortedColIndA, descrC, bsrSortedRowPtrC, rowBlockDim,
                               colBlockDim, nnzTotalDevHostPtr, pBuffer)
    @check @runtime_ccall((:cusparseXcsr2gebsrNnz, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{Cint}, CuPtr{Cint}, cusparseMatDescr_t, CuPtr{Cint}, Cint, Cint,
                  PtrOrCuPtr{Cint}, CuPtr{Cvoid}),
                 handle, dirA, m, n, descrA, csrSortedRowPtrA, csrSortedColIndA, descrC,
                 bsrSortedRowPtrC, rowBlockDim, colBlockDim, nnzTotalDevHostPtr, pBuffer)
end

function cusparseScsr2gebsr(handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA,
                            csrSortedColIndA, descrC, bsrSortedValC, bsrSortedRowPtrC,
                            bsrSortedColIndC, rowBlockDim, colBlockDim, pBuffer)
    @check @runtime_ccall((:cusparseScsr2gebsr, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{Cfloat}, CuPtr{Cint}, CuPtr{Cint}, cusparseMatDescr_t,
                  CuPtr{Cfloat}, CuPtr{Cint}, CuPtr{Cint}, Cint, Cint, CuPtr{Cvoid}),
                 handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA,
                 csrSortedColIndA, descrC, bsrSortedValC, bsrSortedRowPtrC,
                 bsrSortedColIndC, rowBlockDim, colBlockDim, pBuffer)
end

function cusparseDcsr2gebsr(handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA,
                            csrSortedColIndA, descrC, bsrSortedValC, bsrSortedRowPtrC,
                            bsrSortedColIndC, rowBlockDim, colBlockDim, pBuffer)
    @check @runtime_ccall((:cusparseDcsr2gebsr, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{Cdouble}, CuPtr{Cint}, CuPtr{Cint}, cusparseMatDescr_t,
                  CuPtr{Cdouble}, CuPtr{Cint}, CuPtr{Cint}, Cint, Cint, CuPtr{Cvoid}),
                 handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA,
                 csrSortedColIndA, descrC, bsrSortedValC, bsrSortedRowPtrC,
                 bsrSortedColIndC, rowBlockDim, colBlockDim, pBuffer)
end

function cusparseCcsr2gebsr(handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA,
                            csrSortedColIndA, descrC, bsrSortedValC, bsrSortedRowPtrC,
                            bsrSortedColIndC, rowBlockDim, colBlockDim, pBuffer)
    @check @runtime_ccall((:cusparseCcsr2gebsr, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{cuComplex}, CuPtr{Cint}, CuPtr{Cint}, cusparseMatDescr_t,
                  CuPtr{cuComplex}, CuPtr{Cint}, CuPtr{Cint}, Cint, Cint, CuPtr{Cvoid}),
                 handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA,
                 csrSortedColIndA, descrC, bsrSortedValC, bsrSortedRowPtrC,
                 bsrSortedColIndC, rowBlockDim, colBlockDim, pBuffer)
end

function cusparseZcsr2gebsr(handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA,
                            csrSortedColIndA, descrC, bsrSortedValC, bsrSortedRowPtrC,
                            bsrSortedColIndC, rowBlockDim, colBlockDim, pBuffer)
    @check @runtime_ccall((:cusparseZcsr2gebsr, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{cuDoubleComplex}, CuPtr{Cint}, CuPtr{Cint}, cusparseMatDescr_t,
                  CuPtr{cuDoubleComplex}, CuPtr{Cint}, CuPtr{Cint}, Cint, Cint,
                  CuPtr{Cvoid}),
                 handle, dirA, m, n, descrA, csrSortedValA, csrSortedRowPtrA,
                 csrSortedColIndA, descrC, bsrSortedValC, bsrSortedRowPtrC,
                 bsrSortedColIndC, rowBlockDim, colBlockDim, pBuffer)
end

function cusparseSgebsr2gebsr_bufferSize(handle, dirA, mb, nb, nnzb, descrA, bsrSortedValA,
                                         bsrSortedRowPtrA, bsrSortedColIndA, rowBlockDimA,
                                         colBlockDimA, rowBlockDimC, colBlockDimC,
                                         pBufferSizeInBytes)
    @check @runtime_ccall((:cusparseSgebsr2gebsr_bufferSize, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, Cint, Cint, Cint,
                  cusparseMatDescr_t, CuPtr{Cfloat}, CuPtr{Cint}, CuPtr{Cint}, Cint, Cint,
                  Cint, Cint, Ptr{Cint}),
                 handle, dirA, mb, nb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA,
                 bsrSortedColIndA, rowBlockDimA, colBlockDimA, rowBlockDimC, colBlockDimC,
                 pBufferSizeInBytes)
end

function cusparseDgebsr2gebsr_bufferSize(handle, dirA, mb, nb, nnzb, descrA, bsrSortedValA,
                                         bsrSortedRowPtrA, bsrSortedColIndA, rowBlockDimA,
                                         colBlockDimA, rowBlockDimC, colBlockDimC,
                                         pBufferSizeInBytes)
    @check @runtime_ccall((:cusparseDgebsr2gebsr_bufferSize, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, Cint, Cint, Cint,
                  cusparseMatDescr_t, CuPtr{Cdouble}, CuPtr{Cint}, CuPtr{Cint}, Cint, Cint,
                  Cint, Cint, Ptr{Cint}),
                 handle, dirA, mb, nb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA,
                 bsrSortedColIndA, rowBlockDimA, colBlockDimA, rowBlockDimC, colBlockDimC,
                 pBufferSizeInBytes)
end

function cusparseCgebsr2gebsr_bufferSize(handle, dirA, mb, nb, nnzb, descrA, bsrSortedValA,
                                         bsrSortedRowPtrA, bsrSortedColIndA, rowBlockDimA,
                                         colBlockDimA, rowBlockDimC, colBlockDimC,
                                         pBufferSizeInBytes)
    @check @runtime_ccall((:cusparseCgebsr2gebsr_bufferSize, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, Cint, Cint, Cint,
                  cusparseMatDescr_t, CuPtr{cuComplex}, CuPtr{Cint}, CuPtr{Cint}, Cint,
                  Cint, Cint, Cint, Ptr{Cint}),
                 handle, dirA, mb, nb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA,
                 bsrSortedColIndA, rowBlockDimA, colBlockDimA, rowBlockDimC, colBlockDimC,
                 pBufferSizeInBytes)
end

function cusparseZgebsr2gebsr_bufferSize(handle, dirA, mb, nb, nnzb, descrA, bsrSortedValA,
                                         bsrSortedRowPtrA, bsrSortedColIndA, rowBlockDimA,
                                         colBlockDimA, rowBlockDimC, colBlockDimC,
                                         pBufferSizeInBytes)
    @check @runtime_ccall((:cusparseZgebsr2gebsr_bufferSize, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, Cint, Cint, Cint,
                  cusparseMatDescr_t, CuPtr{cuDoubleComplex}, CuPtr{Cint}, CuPtr{Cint},
                  Cint, Cint, Cint, Cint, Ptr{Cint}),
                 handle, dirA, mb, nb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA,
                 bsrSortedColIndA, rowBlockDimA, colBlockDimA, rowBlockDimC, colBlockDimC,
                 pBufferSizeInBytes)
end

function cusparseSgebsr2gebsr_bufferSizeExt(handle, dirA, mb, nb, nnzb, descrA,
                                            bsrSortedValA, bsrSortedRowPtrA,
                                            bsrSortedColIndA, rowBlockDimA, colBlockDimA,
                                            rowBlockDimC, colBlockDimC, pBufferSize)
    @check @runtime_ccall((:cusparseSgebsr2gebsr_bufferSizeExt, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, Cint, Cint, Cint,
                  cusparseMatDescr_t, CuPtr{Cfloat}, CuPtr{Cint}, CuPtr{Cint}, Cint, Cint,
                  Cint, Cint, Ptr{Csize_t}),
                 handle, dirA, mb, nb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA,
                 bsrSortedColIndA, rowBlockDimA, colBlockDimA, rowBlockDimC, colBlockDimC,
                 pBufferSize)
end

function cusparseDgebsr2gebsr_bufferSizeExt(handle, dirA, mb, nb, nnzb, descrA,
                                            bsrSortedValA, bsrSortedRowPtrA,
                                            bsrSortedColIndA, rowBlockDimA, colBlockDimA,
                                            rowBlockDimC, colBlockDimC, pBufferSize)
    @check @runtime_ccall((:cusparseDgebsr2gebsr_bufferSizeExt, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, Cint, Cint, Cint,
                  cusparseMatDescr_t, CuPtr{Cdouble}, CuPtr{Cint}, CuPtr{Cint}, Cint, Cint,
                  Cint, Cint, Ptr{Csize_t}),
                 handle, dirA, mb, nb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA,
                 bsrSortedColIndA, rowBlockDimA, colBlockDimA, rowBlockDimC, colBlockDimC,
                 pBufferSize)
end

function cusparseCgebsr2gebsr_bufferSizeExt(handle, dirA, mb, nb, nnzb, descrA,
                                            bsrSortedValA, bsrSortedRowPtrA,
                                            bsrSortedColIndA, rowBlockDimA, colBlockDimA,
                                            rowBlockDimC, colBlockDimC, pBufferSize)
    @check @runtime_ccall((:cusparseCgebsr2gebsr_bufferSizeExt, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, Cint, Cint, Cint,
                  cusparseMatDescr_t, CuPtr{cuComplex}, CuPtr{Cint}, CuPtr{Cint}, Cint,
                  Cint, Cint, Cint, Ptr{Csize_t}),
                 handle, dirA, mb, nb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA,
                 bsrSortedColIndA, rowBlockDimA, colBlockDimA, rowBlockDimC, colBlockDimC,
                 pBufferSize)
end

function cusparseZgebsr2gebsr_bufferSizeExt(handle, dirA, mb, nb, nnzb, descrA,
                                            bsrSortedValA, bsrSortedRowPtrA,
                                            bsrSortedColIndA, rowBlockDimA, colBlockDimA,
                                            rowBlockDimC, colBlockDimC, pBufferSize)
    @check @runtime_ccall((:cusparseZgebsr2gebsr_bufferSizeExt, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, Cint, Cint, Cint,
                  cusparseMatDescr_t, CuPtr{cuDoubleComplex}, CuPtr{Cint}, CuPtr{Cint},
                  Cint, Cint, Cint, Cint, Ptr{Csize_t}),
                 handle, dirA, mb, nb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA,
                 bsrSortedColIndA, rowBlockDimA, colBlockDimA, rowBlockDimC, colBlockDimC,
                 pBufferSize)
end

function cusparseXgebsr2gebsrNnz(handle, dirA, mb, nb, nnzb, descrA, bsrSortedRowPtrA,
                                 bsrSortedColIndA, rowBlockDimA, colBlockDimA, descrC,
                                 bsrSortedRowPtrC, rowBlockDimC, colBlockDimC,
                                 nnzTotalDevHostPtr, pBuffer)
    @check @runtime_ccall((:cusparseXgebsr2gebsrNnz, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, Cint, Cint, Cint,
                  cusparseMatDescr_t, CuPtr{Cint}, CuPtr{Cint}, Cint, Cint,
                  cusparseMatDescr_t, CuPtr{Cint}, Cint, Cint, PtrOrCuPtr{Cint},
                  CuPtr{Cvoid}),
                 handle, dirA, mb, nb, nnzb, descrA, bsrSortedRowPtrA, bsrSortedColIndA,
                 rowBlockDimA, colBlockDimA, descrC, bsrSortedRowPtrC, rowBlockDimC,
                 colBlockDimC, nnzTotalDevHostPtr, pBuffer)
end

function cusparseSgebsr2gebsr(handle, dirA, mb, nb, nnzb, descrA, bsrSortedValA,
                              bsrSortedRowPtrA, bsrSortedColIndA, rowBlockDimA,
                              colBlockDimA, descrC, bsrSortedValC, bsrSortedRowPtrC,
                              bsrSortedColIndC, rowBlockDimC, colBlockDimC, pBuffer)
    @check @runtime_ccall((:cusparseSgebsr2gebsr, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, Cint, Cint, Cint,
                  cusparseMatDescr_t, CuPtr{Cfloat}, CuPtr{Cint}, CuPtr{Cint}, Cint, Cint,
                  cusparseMatDescr_t, CuPtr{Cfloat}, CuPtr{Cint}, CuPtr{Cint}, Cint, Cint,
                  CuPtr{Cvoid}),
                 handle, dirA, mb, nb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA,
                 bsrSortedColIndA, rowBlockDimA, colBlockDimA, descrC, bsrSortedValC,
                 bsrSortedRowPtrC, bsrSortedColIndC, rowBlockDimC, colBlockDimC, pBuffer)
end

function cusparseDgebsr2gebsr(handle, dirA, mb, nb, nnzb, descrA, bsrSortedValA,
                              bsrSortedRowPtrA, bsrSortedColIndA, rowBlockDimA,
                              colBlockDimA, descrC, bsrSortedValC, bsrSortedRowPtrC,
                              bsrSortedColIndC, rowBlockDimC, colBlockDimC, pBuffer)
    @check @runtime_ccall((:cusparseDgebsr2gebsr, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, Cint, Cint, Cint,
                  cusparseMatDescr_t, CuPtr{Cdouble}, CuPtr{Cint}, CuPtr{Cint}, Cint, Cint,
                  cusparseMatDescr_t, CuPtr{Cdouble}, CuPtr{Cint}, CuPtr{Cint}, Cint, Cint,
                  CuPtr{Cvoid}),
                 handle, dirA, mb, nb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA,
                 bsrSortedColIndA, rowBlockDimA, colBlockDimA, descrC, bsrSortedValC,
                 bsrSortedRowPtrC, bsrSortedColIndC, rowBlockDimC, colBlockDimC, pBuffer)
end

function cusparseCgebsr2gebsr(handle, dirA, mb, nb, nnzb, descrA, bsrSortedValA,
                              bsrSortedRowPtrA, bsrSortedColIndA, rowBlockDimA,
                              colBlockDimA, descrC, bsrSortedValC, bsrSortedRowPtrC,
                              bsrSortedColIndC, rowBlockDimC, colBlockDimC, pBuffer)
    @check @runtime_ccall((:cusparseCgebsr2gebsr, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, Cint, Cint, Cint,
                  cusparseMatDescr_t, CuPtr{cuComplex}, CuPtr{Cint}, CuPtr{Cint}, Cint,
                  Cint, cusparseMatDescr_t, CuPtr{cuComplex}, CuPtr{Cint}, CuPtr{Cint},
                  Cint, Cint, CuPtr{Cvoid}),
                 handle, dirA, mb, nb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA,
                 bsrSortedColIndA, rowBlockDimA, colBlockDimA, descrC, bsrSortedValC,
                 bsrSortedRowPtrC, bsrSortedColIndC, rowBlockDimC, colBlockDimC, pBuffer)
end

function cusparseZgebsr2gebsr(handle, dirA, mb, nb, nnzb, descrA, bsrSortedValA,
                              bsrSortedRowPtrA, bsrSortedColIndA, rowBlockDimA,
                              colBlockDimA, descrC, bsrSortedValC, bsrSortedRowPtrC,
                              bsrSortedColIndC, rowBlockDimC, colBlockDimC, pBuffer)
    @check @runtime_ccall((:cusparseZgebsr2gebsr, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseDirection_t, Cint, Cint, Cint,
                  cusparseMatDescr_t, CuPtr{cuDoubleComplex}, CuPtr{Cint}, CuPtr{Cint},
                  Cint, Cint, cusparseMatDescr_t, CuPtr{cuDoubleComplex}, CuPtr{Cint},
                  CuPtr{Cint}, Cint, Cint, CuPtr{Cvoid}),
                 handle, dirA, mb, nb, nnzb, descrA, bsrSortedValA, bsrSortedRowPtrA,
                 bsrSortedColIndA, rowBlockDimA, colBlockDimA, descrC, bsrSortedValC,
                 bsrSortedRowPtrC, bsrSortedColIndC, rowBlockDimC, colBlockDimC, pBuffer)
end

function cusparseCreateIdentityPermutation(handle, n, p)
    @check @runtime_ccall((:cusparseCreateIdentityPermutation, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, CuPtr{Cint}),
                 handle, n, p)
end

function cusparseXcoosort_bufferSizeExt(handle, m, n, nnz, cooRowsA, cooColsA,
                                        pBufferSizeInBytes)
    @check @runtime_ccall((:cusparseXcoosort_bufferSizeExt, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, Cint, CuPtr{Cint}, CuPtr{Cint},
                  Ptr{Csize_t}),
                 handle, m, n, nnz, cooRowsA, cooColsA, pBufferSizeInBytes)
end

function cusparseXcoosortByRow(handle, m, n, nnz, cooRowsA, cooColsA, P, pBuffer)
    @check @runtime_ccall((:cusparseXcoosortByRow, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, Cint, CuPtr{Cint}, CuPtr{Cint},
                  CuPtr{Cint}, CuPtr{Cvoid}),
                 handle, m, n, nnz, cooRowsA, cooColsA, P, pBuffer)
end

function cusparseXcoosortByColumn(handle, m, n, nnz, cooRowsA, cooColsA, P, pBuffer)
    @check @runtime_ccall((:cusparseXcoosortByColumn, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, Cint, CuPtr{Cint}, CuPtr{Cint},
                  CuPtr{Cint}, CuPtr{Cvoid}),
                 handle, m, n, nnz, cooRowsA, cooColsA, P, pBuffer)
end

function cusparseXcsrsort_bufferSizeExt(handle, m, n, nnz, csrRowPtrA, csrColIndA,
                                        pBufferSizeInBytes)
    @check @runtime_ccall((:cusparseXcsrsort_bufferSizeExt, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, Cint, CuPtr{Cint}, CuPtr{Cint},
                  Ptr{Csize_t}),
                 handle, m, n, nnz, csrRowPtrA, csrColIndA, pBufferSizeInBytes)
end

function cusparseXcsrsort(handle, m, n, nnz, descrA, csrRowPtrA, csrColIndA, P, pBuffer)
    @check @runtime_ccall((:cusparseXcsrsort, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, Cint, cusparseMatDescr_t, CuPtr{Cint},
                  CuPtr{Cint}, CuPtr{Cint}, CuPtr{Cvoid}),
                 handle, m, n, nnz, descrA, csrRowPtrA, csrColIndA, P, pBuffer)
end

function cusparseXcscsort_bufferSizeExt(handle, m, n, nnz, cscColPtrA, cscRowIndA,
                                        pBufferSizeInBytes)
    @check @runtime_ccall((:cusparseXcscsort_bufferSizeExt, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, Cint, CuPtr{Cint}, CuPtr{Cint},
                  Ptr{Csize_t}),
                 handle, m, n, nnz, cscColPtrA, cscRowIndA, pBufferSizeInBytes)
end

function cusparseXcscsort(handle, m, n, nnz, descrA, cscColPtrA, cscRowIndA, P, pBuffer)
    @check @runtime_ccall((:cusparseXcscsort, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, Cint, cusparseMatDescr_t, CuPtr{Cint},
                  CuPtr{Cint}, CuPtr{Cint}, CuPtr{Cvoid}),
                 handle, m, n, nnz, descrA, cscColPtrA, cscRowIndA, P, pBuffer)
end

function cusparseScsru2csr_bufferSizeExt(handle, m, n, nnz, csrVal, csrRowPtr, csrColInd,
                                         info, pBufferSizeInBytes)
    @check @runtime_ccall((:cusparseScsru2csr_bufferSizeExt, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, Cint, CuPtr{Cfloat}, CuPtr{Cint},
                  CuPtr{Cint}, csru2csrInfo_t, Ptr{Csize_t}),
                 handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, info, pBufferSizeInBytes)
end

function cusparseDcsru2csr_bufferSizeExt(handle, m, n, nnz, csrVal, csrRowPtr, csrColInd,
                                         info, pBufferSizeInBytes)
    @check @runtime_ccall((:cusparseDcsru2csr_bufferSizeExt, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, Cint, CuPtr{Cdouble}, CuPtr{Cint},
                  CuPtr{Cint}, csru2csrInfo_t, Ptr{Csize_t}),
                 handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, info, pBufferSizeInBytes)
end

function cusparseCcsru2csr_bufferSizeExt(handle, m, n, nnz, csrVal, csrRowPtr, csrColInd,
                                         info, pBufferSizeInBytes)
    @check @runtime_ccall((:cusparseCcsru2csr_bufferSizeExt, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, Cint, CuPtr{cuComplex}, CuPtr{Cint},
                  CuPtr{Cint}, csru2csrInfo_t, Ptr{Csize_t}),
                 handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, info, pBufferSizeInBytes)
end

function cusparseZcsru2csr_bufferSizeExt(handle, m, n, nnz, csrVal, csrRowPtr, csrColInd,
                                         info, pBufferSizeInBytes)
    @check @runtime_ccall((:cusparseZcsru2csr_bufferSizeExt, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, Cint, CuPtr{cuDoubleComplex}, CuPtr{Cint},
                  CuPtr{Cint}, csru2csrInfo_t, Ptr{Csize_t}),
                 handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, info, pBufferSizeInBytes)
end

function cusparseScsru2csr(handle, m, n, nnz, descrA, csrVal, csrRowPtr, csrColInd, info,
                           pBuffer)
    @check @runtime_ccall((:cusparseScsru2csr, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, Cint, cusparseMatDescr_t, CuPtr{Cfloat},
                  CuPtr{Cint}, CuPtr{Cint}, csru2csrInfo_t, CuPtr{Cvoid}),
                 handle, m, n, nnz, descrA, csrVal, csrRowPtr, csrColInd, info, pBuffer)
end

function cusparseDcsru2csr(handle, m, n, nnz, descrA, csrVal, csrRowPtr, csrColInd, info,
                           pBuffer)
    @check @runtime_ccall((:cusparseDcsru2csr, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, Cint, cusparseMatDescr_t, CuPtr{Cdouble},
                  CuPtr{Cint}, CuPtr{Cint}, csru2csrInfo_t, CuPtr{Cvoid}),
                 handle, m, n, nnz, descrA, csrVal, csrRowPtr, csrColInd, info, pBuffer)
end

function cusparseCcsru2csr(handle, m, n, nnz, descrA, csrVal, csrRowPtr, csrColInd, info,
                           pBuffer)
    @check @runtime_ccall((:cusparseCcsru2csr, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{cuComplex}, CuPtr{Cint}, CuPtr{Cint}, csru2csrInfo_t, CuPtr{Cvoid}),
                 handle, m, n, nnz, descrA, csrVal, csrRowPtr, csrColInd, info, pBuffer)
end

function cusparseZcsru2csr(handle, m, n, nnz, descrA, csrVal, csrRowPtr, csrColInd, info,
                           pBuffer)
    @check @runtime_ccall((:cusparseZcsru2csr, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{cuDoubleComplex}, CuPtr{Cint}, CuPtr{Cint}, csru2csrInfo_t,
                  CuPtr{Cvoid}),
                 handle, m, n, nnz, descrA, csrVal, csrRowPtr, csrColInd, info, pBuffer)
end

function cusparseScsr2csru(handle, m, n, nnz, descrA, csrVal, csrRowPtr, csrColInd, info,
                           pBuffer)
    @check @runtime_ccall((:cusparseScsr2csru, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, Cint, cusparseMatDescr_t, CuPtr{Cfloat},
                  CuPtr{Cint}, CuPtr{Cint}, csru2csrInfo_t, CuPtr{Cvoid}),
                 handle, m, n, nnz, descrA, csrVal, csrRowPtr, csrColInd, info, pBuffer)
end

function cusparseDcsr2csru(handle, m, n, nnz, descrA, csrVal, csrRowPtr, csrColInd, info,
                           pBuffer)
    @check @runtime_ccall((:cusparseDcsr2csru, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, Cint, cusparseMatDescr_t, CuPtr{Cdouble},
                  CuPtr{Cint}, CuPtr{Cint}, csru2csrInfo_t, CuPtr{Cvoid}),
                 handle, m, n, nnz, descrA, csrVal, csrRowPtr, csrColInd, info, pBuffer)
end

function cusparseCcsr2csru(handle, m, n, nnz, descrA, csrVal, csrRowPtr, csrColInd, info,
                           pBuffer)
    @check @runtime_ccall((:cusparseCcsr2csru, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{cuComplex}, CuPtr{Cint}, CuPtr{Cint}, csru2csrInfo_t, CuPtr{Cvoid}),
                 handle, m, n, nnz, descrA, csrVal, csrRowPtr, csrColInd, info, pBuffer)
end

function cusparseZcsr2csru(handle, m, n, nnz, descrA, csrVal, csrRowPtr, csrColInd, info,
                           pBuffer)
    @check @runtime_ccall((:cusparseZcsr2csru, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, Cint, cusparseMatDescr_t,
                  CuPtr{cuDoubleComplex}, CuPtr{Cint}, CuPtr{Cint}, csru2csrInfo_t,
                  CuPtr{Cvoid}),
                 handle, m, n, nnz, descrA, csrVal, csrRowPtr, csrColInd, info, pBuffer)
end

function cusparseSpruneDense2csr_bufferSizeExt(handle, m, n, A, lda, threshold, descrC,
                                               csrSortedValC, csrSortedRowPtrC,
                                               csrSortedColIndC, pBufferSizeInBytes)
    @check @runtime_ccall((:cusparseSpruneDense2csr_bufferSizeExt, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, CuPtr{Cfloat}, Cint, Ptr{Cfloat},
                  cusparseMatDescr_t, CuPtr{Cfloat}, CuPtr{Cint}, CuPtr{Cint}, Ptr{Csize_t}),
                 handle, m, n, A, lda, threshold, descrC, csrSortedValC, csrSortedRowPtrC,
                 csrSortedColIndC, pBufferSizeInBytes)
end

function cusparseDpruneDense2csr_bufferSizeExt(handle, m, n, A, lda, threshold, descrC,
                                               csrSortedValC, csrSortedRowPtrC,
                                               csrSortedColIndC, pBufferSizeInBytes)
    @check @runtime_ccall((:cusparseDpruneDense2csr_bufferSizeExt, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, CuPtr{Cdouble}, Cint, Ptr{Cdouble},
                  cusparseMatDescr_t, CuPtr{Cdouble}, CuPtr{Cint}, CuPtr{Cint},
                  Ptr{Csize_t}),
                 handle, m, n, A, lda, threshold, descrC, csrSortedValC, csrSortedRowPtrC,
                 csrSortedColIndC, pBufferSizeInBytes)
end

function cusparseSpruneDense2csrNnz(handle, m, n, A, lda, threshold, descrC, csrRowPtrC,
                                    nnzTotalDevHostPtr, pBuffer)
    @check @runtime_ccall((:cusparseSpruneDense2csrNnz, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, CuPtr{Cfloat}, Cint, Ptr{Cfloat},
                  cusparseMatDescr_t, CuPtr{Cint}, PtrOrCuPtr{Cint}, CuPtr{Cvoid}),
                 handle, m, n, A, lda, threshold, descrC, csrRowPtrC, nnzTotalDevHostPtr,
                 pBuffer)
end

function cusparseDpruneDense2csrNnz(handle, m, n, A, lda, threshold, descrC,
                                    csrSortedRowPtrC, nnzTotalDevHostPtr, pBuffer)
    @check @runtime_ccall((:cusparseDpruneDense2csrNnz, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, CuPtr{Cdouble}, Cint, Ptr{Cdouble},
                  cusparseMatDescr_t, CuPtr{Cint}, PtrOrCuPtr{Cint}, CuPtr{Cvoid}),
                 handle, m, n, A, lda, threshold, descrC, csrSortedRowPtrC,
                 nnzTotalDevHostPtr, pBuffer)
end

function cusparseSpruneDense2csr(handle, m, n, A, lda, threshold, descrC, csrSortedValC,
                                 csrSortedRowPtrC, csrSortedColIndC, pBuffer)
    @check @runtime_ccall((:cusparseSpruneDense2csr, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, CuPtr{Cfloat}, Cint, Ptr{Cfloat},
                  cusparseMatDescr_t, CuPtr{Cfloat}, CuPtr{Cint}, CuPtr{Cint}, CuPtr{Cvoid}),
                 handle, m, n, A, lda, threshold, descrC, csrSortedValC, csrSortedRowPtrC,
                 csrSortedColIndC, pBuffer)
end

function cusparseDpruneDense2csr(handle, m, n, A, lda, threshold, descrC, csrSortedValC,
                                 csrSortedRowPtrC, csrSortedColIndC, pBuffer)
    @check @runtime_ccall((:cusparseDpruneDense2csr, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, CuPtr{Cdouble}, Cint, Ptr{Cdouble},
                  cusparseMatDescr_t, CuPtr{Cdouble}, CuPtr{Cint}, CuPtr{Cint},
                  CuPtr{Cvoid}),
                 handle, m, n, A, lda, threshold, descrC, csrSortedValC, csrSortedRowPtrC,
                 csrSortedColIndC, pBuffer)
end

function cusparseSpruneCsr2csr_bufferSizeExt(handle, m, n, nnzA, descrA, csrSortedValA,
                                             csrSortedRowPtrA, csrSortedColIndA, threshold,
                                             descrC, csrSortedValC, csrSortedRowPtrC,
                                             csrSortedColIndC, pBufferSizeInBytes)
    @check @runtime_ccall((:cusparseSpruneCsr2csr_bufferSizeExt, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, Cint, cusparseMatDescr_t, CuPtr{Cfloat},
                  CuPtr{Cint}, CuPtr{Cint}, Ptr{Cfloat}, cusparseMatDescr_t, CuPtr{Cfloat},
                  CuPtr{Cint}, CuPtr{Cint}, Ptr{Csize_t}),
                 handle, m, n, nnzA, descrA, csrSortedValA, csrSortedRowPtrA,
                 csrSortedColIndA, threshold, descrC, csrSortedValC, csrSortedRowPtrC,
                 csrSortedColIndC, pBufferSizeInBytes)
end

function cusparseDpruneCsr2csr_bufferSizeExt(handle, m, n, nnzA, descrA, csrSortedValA,
                                             csrSortedRowPtrA, csrSortedColIndA, threshold,
                                             descrC, csrSortedValC, csrSortedRowPtrC,
                                             csrSortedColIndC, pBufferSizeInBytes)
    @check @runtime_ccall((:cusparseDpruneCsr2csr_bufferSizeExt, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, Cint, cusparseMatDescr_t, CuPtr{Cdouble},
                  CuPtr{Cint}, CuPtr{Cint}, Ptr{Cdouble}, cusparseMatDescr_t,
                  CuPtr{Cdouble}, CuPtr{Cint}, CuPtr{Cint}, Ptr{Csize_t}),
                 handle, m, n, nnzA, descrA, csrSortedValA, csrSortedRowPtrA,
                 csrSortedColIndA, threshold, descrC, csrSortedValC, csrSortedRowPtrC,
                 csrSortedColIndC, pBufferSizeInBytes)
end

function cusparseSpruneCsr2csrNnz(handle, m, n, nnzA, descrA, csrSortedValA,
                                  csrSortedRowPtrA, csrSortedColIndA, threshold, descrC,
                                  csrSortedRowPtrC, nnzTotalDevHostPtr, pBuffer)
    @check @runtime_ccall((:cusparseSpruneCsr2csrNnz, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, Cint, cusparseMatDescr_t, CuPtr{Cfloat},
                  CuPtr{Cint}, CuPtr{Cint}, Ptr{Cfloat}, cusparseMatDescr_t, CuPtr{Cint},
                  PtrOrCuPtr{Cint}, CuPtr{Cvoid}),
                 handle, m, n, nnzA, descrA, csrSortedValA, csrSortedRowPtrA,
                 csrSortedColIndA, threshold, descrC, csrSortedRowPtrC, nnzTotalDevHostPtr,
                 pBuffer)
end

function cusparseDpruneCsr2csrNnz(handle, m, n, nnzA, descrA, csrSortedValA,
                                  csrSortedRowPtrA, csrSortedColIndA, threshold, descrC,
                                  csrSortedRowPtrC, nnzTotalDevHostPtr, pBuffer)
    @check @runtime_ccall((:cusparseDpruneCsr2csrNnz, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, Cint, cusparseMatDescr_t, CuPtr{Cdouble},
                  CuPtr{Cint}, CuPtr{Cint}, Ptr{Cdouble}, cusparseMatDescr_t, CuPtr{Cint},
                  PtrOrCuPtr{Cint}, CuPtr{Cvoid}),
                 handle, m, n, nnzA, descrA, csrSortedValA, csrSortedRowPtrA,
                 csrSortedColIndA, threshold, descrC, csrSortedRowPtrC, nnzTotalDevHostPtr,
                 pBuffer)
end

function cusparseSpruneCsr2csr(handle, m, n, nnzA, descrA, csrSortedValA, csrSortedRowPtrA,
                               csrSortedColIndA, threshold, descrC, csrSortedValC,
                               csrSortedRowPtrC, csrSortedColIndC, pBuffer)
    @check @runtime_ccall((:cusparseSpruneCsr2csr, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, Cint, cusparseMatDescr_t, CuPtr{Cfloat},
                  CuPtr{Cint}, CuPtr{Cint}, Ptr{Cfloat}, cusparseMatDescr_t, CuPtr{Cfloat},
                  CuPtr{Cint}, CuPtr{Cint}, CuPtr{Cvoid}),
                 handle, m, n, nnzA, descrA, csrSortedValA, csrSortedRowPtrA,
                 csrSortedColIndA, threshold, descrC, csrSortedValC, csrSortedRowPtrC,
                 csrSortedColIndC, pBuffer)
end

function cusparseDpruneCsr2csr(handle, m, n, nnzA, descrA, csrSortedValA, csrSortedRowPtrA,
                               csrSortedColIndA, threshold, descrC, csrSortedValC,
                               csrSortedRowPtrC, csrSortedColIndC, pBuffer)
    @check @runtime_ccall((:cusparseDpruneCsr2csr, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, Cint, cusparseMatDescr_t, CuPtr{Cdouble},
                  CuPtr{Cint}, CuPtr{Cint}, Ptr{Cdouble}, cusparseMatDescr_t,
                  CuPtr{Cdouble}, CuPtr{Cint}, CuPtr{Cint}, CuPtr{Cvoid}),
                 handle, m, n, nnzA, descrA, csrSortedValA, csrSortedRowPtrA,
                 csrSortedColIndA, threshold, descrC, csrSortedValC, csrSortedRowPtrC,
                 csrSortedColIndC, pBuffer)
end

function cusparseSpruneDense2csrByPercentage_bufferSizeExt(handle, m, n, A, lda,
                                                           percentage, descrC,
                                                           csrSortedValC, csrSortedRowPtrC,
                                                           csrSortedColIndC, info,
                                                           pBufferSizeInBytes)
    @check @runtime_ccall((:cusparseSpruneDense2csrByPercentage_bufferSizeExt, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, CuPtr{Cfloat}, Cint, Cfloat,
                  cusparseMatDescr_t, CuPtr{Cfloat}, CuPtr{Cint}, CuPtr{Cint}, pruneInfo_t,
                  Ptr{Csize_t}),
                 handle, m, n, A, lda, percentage, descrC, csrSortedValC, csrSortedRowPtrC,
                 csrSortedColIndC, info, pBufferSizeInBytes)
end

function cusparseDpruneDense2csrByPercentage_bufferSizeExt(handle, m, n, A, lda,
                                                           percentage, descrC,
                                                           csrSortedValC, csrSortedRowPtrC,
                                                           csrSortedColIndC, info,
                                                           pBufferSizeInBytes)
    @check @runtime_ccall((:cusparseDpruneDense2csrByPercentage_bufferSizeExt, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, CuPtr{Cdouble}, Cint, Cfloat,
                  cusparseMatDescr_t, CuPtr{Cdouble}, CuPtr{Cint}, CuPtr{Cint},
                  pruneInfo_t, Ptr{Csize_t}),
                 handle, m, n, A, lda, percentage, descrC, csrSortedValC, csrSortedRowPtrC,
                 csrSortedColIndC, info, pBufferSizeInBytes)
end

function cusparseSpruneDense2csrNnzByPercentage(handle, m, n, A, lda, percentage, descrC,
                                                csrRowPtrC, nnzTotalDevHostPtr, info,
                                                pBuffer)
    @check @runtime_ccall((:cusparseSpruneDense2csrNnzByPercentage, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, CuPtr{Cfloat}, Cint, Cfloat,
                  cusparseMatDescr_t, CuPtr{Cint}, PtrOrCuPtr{Cint}, pruneInfo_t,
                  CuPtr{Cvoid}),
                 handle, m, n, A, lda, percentage, descrC, csrRowPtrC, nnzTotalDevHostPtr,
                 info, pBuffer)
end

function cusparseDpruneDense2csrNnzByPercentage(handle, m, n, A, lda, percentage, descrC,
                                                csrRowPtrC, nnzTotalDevHostPtr, info,
                                                pBuffer)
    @check @runtime_ccall((:cusparseDpruneDense2csrNnzByPercentage, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, CuPtr{Cdouble}, Cint, Cfloat,
                  cusparseMatDescr_t, CuPtr{Cint}, PtrOrCuPtr{Cint}, pruneInfo_t,
                  CuPtr{Cvoid}),
                 handle, m, n, A, lda, percentage, descrC, csrRowPtrC, nnzTotalDevHostPtr,
                 info, pBuffer)
end

function cusparseSpruneDense2csrByPercentage(handle, m, n, A, lda, percentage, descrC,
                                             csrSortedValC, csrSortedRowPtrC,
                                             csrSortedColIndC, info, pBuffer)
    @check @runtime_ccall((:cusparseSpruneDense2csrByPercentage, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, CuPtr{Cfloat}, Cint, Cfloat,
                  cusparseMatDescr_t, CuPtr{Cfloat}, CuPtr{Cint}, CuPtr{Cint}, pruneInfo_t,
                  CuPtr{Cvoid}),
                 handle, m, n, A, lda, percentage, descrC, csrSortedValC, csrSortedRowPtrC,
                 csrSortedColIndC, info, pBuffer)
end

function cusparseDpruneDense2csrByPercentage(handle, m, n, A, lda, percentage, descrC,
                                             csrSortedValC, csrSortedRowPtrC,
                                             csrSortedColIndC, info, pBuffer)
    @check @runtime_ccall((:cusparseDpruneDense2csrByPercentage, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, CuPtr{Cdouble}, Cint, Cfloat,
                  cusparseMatDescr_t, CuPtr{Cdouble}, CuPtr{Cint}, CuPtr{Cint},
                  pruneInfo_t, CuPtr{Cvoid}),
                 handle, m, n, A, lda, percentage, descrC, csrSortedValC, csrSortedRowPtrC,
                 csrSortedColIndC, info, pBuffer)
end

function cusparseSpruneCsr2csrByPercentage_bufferSizeExt(handle, m, n, nnzA, descrA,
                                                         csrSortedValA, csrSortedRowPtrA,
                                                         csrSortedColIndA, percentage,
                                                         descrC, csrSortedValC,
                                                         csrSortedRowPtrC,
                                                         csrSortedColIndC, info,
                                                         pBufferSizeInBytes)
    @check @runtime_ccall((:cusparseSpruneCsr2csrByPercentage_bufferSizeExt, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, Cint, cusparseMatDescr_t, CuPtr{Cfloat},
                  CuPtr{Cint}, CuPtr{Cint}, Cfloat, cusparseMatDescr_t, CuPtr{Cfloat},
                  CuPtr{Cint}, CuPtr{Cint}, pruneInfo_t, Ptr{Csize_t}),
                 handle, m, n, nnzA, descrA, csrSortedValA, csrSortedRowPtrA,
                 csrSortedColIndA, percentage, descrC, csrSortedValC, csrSortedRowPtrC,
                 csrSortedColIndC, info, pBufferSizeInBytes)
end

function cusparseDpruneCsr2csrByPercentage_bufferSizeExt(handle, m, n, nnzA, descrA,
                                                         csrSortedValA, csrSortedRowPtrA,
                                                         csrSortedColIndA, percentage,
                                                         descrC, csrSortedValC,
                                                         csrSortedRowPtrC,
                                                         csrSortedColIndC, info,
                                                         pBufferSizeInBytes)
    @check @runtime_ccall((:cusparseDpruneCsr2csrByPercentage_bufferSizeExt, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, Cint, cusparseMatDescr_t, CuPtr{Cdouble},
                  CuPtr{Cint}, CuPtr{Cint}, Cfloat, cusparseMatDescr_t, CuPtr{Cdouble},
                  CuPtr{Cint}, CuPtr{Cint}, pruneInfo_t, Ptr{Csize_t}),
                 handle, m, n, nnzA, descrA, csrSortedValA, csrSortedRowPtrA,
                 csrSortedColIndA, percentage, descrC, csrSortedValC, csrSortedRowPtrC,
                 csrSortedColIndC, info, pBufferSizeInBytes)
end

function cusparseSpruneCsr2csrNnzByPercentage(handle, m, n, nnzA, descrA, csrSortedValA,
                                              csrSortedRowPtrA, csrSortedColIndA,
                                              percentage, descrC, csrSortedRowPtrC,
                                              nnzTotalDevHostPtr, info, pBuffer)
    @check @runtime_ccall((:cusparseSpruneCsr2csrNnzByPercentage, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, Cint, cusparseMatDescr_t, CuPtr{Cfloat},
                  CuPtr{Cint}, CuPtr{Cint}, Cfloat, cusparseMatDescr_t, CuPtr{Cint},
                  PtrOrCuPtr{Cint}, pruneInfo_t, CuPtr{Cvoid}),
                 handle, m, n, nnzA, descrA, csrSortedValA, csrSortedRowPtrA,
                 csrSortedColIndA, percentage, descrC, csrSortedRowPtrC,
                 nnzTotalDevHostPtr, info, pBuffer)
end

function cusparseDpruneCsr2csrNnzByPercentage(handle, m, n, nnzA, descrA, csrSortedValA,
                                              csrSortedRowPtrA, csrSortedColIndA,
                                              percentage, descrC, csrSortedRowPtrC,
                                              nnzTotalDevHostPtr, info, pBuffer)
    @check @runtime_ccall((:cusparseDpruneCsr2csrNnzByPercentage, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, Cint, cusparseMatDescr_t, CuPtr{Cdouble},
                  CuPtr{Cint}, CuPtr{Cint}, Cfloat, cusparseMatDescr_t, CuPtr{Cint},
                  PtrOrCuPtr{Cint}, pruneInfo_t, CuPtr{Cvoid}),
                 handle, m, n, nnzA, descrA, csrSortedValA, csrSortedRowPtrA,
                 csrSortedColIndA, percentage, descrC, csrSortedRowPtrC,
                 nnzTotalDevHostPtr, info, pBuffer)
end

function cusparseSpruneCsr2csrByPercentage(handle, m, n, nnzA, descrA, csrSortedValA,
                                           csrSortedRowPtrA, csrSortedColIndA, percentage,
                                           descrC, csrSortedValC, csrSortedRowPtrC,
                                           csrSortedColIndC, info, pBuffer)
    @check @runtime_ccall((:cusparseSpruneCsr2csrByPercentage, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, Cint, cusparseMatDescr_t, CuPtr{Cfloat},
                  CuPtr{Cint}, CuPtr{Cint}, Cfloat, cusparseMatDescr_t, CuPtr{Cfloat},
                  CuPtr{Cint}, CuPtr{Cint}, pruneInfo_t, CuPtr{Cvoid}),
                 handle, m, n, nnzA, descrA, csrSortedValA, csrSortedRowPtrA,
                 csrSortedColIndA, percentage, descrC, csrSortedValC, csrSortedRowPtrC,
                 csrSortedColIndC, info, pBuffer)
end

function cusparseDpruneCsr2csrByPercentage(handle, m, n, nnzA, descrA, csrSortedValA,
                                           csrSortedRowPtrA, csrSortedColIndA, percentage,
                                           descrC, csrSortedValC, csrSortedRowPtrC,
                                           csrSortedColIndC, info, pBuffer)
    @check @runtime_ccall((:cusparseDpruneCsr2csrByPercentage, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, Cint, cusparseMatDescr_t, CuPtr{Cdouble},
                  CuPtr{Cint}, CuPtr{Cint}, Cfloat, cusparseMatDescr_t, CuPtr{Cdouble},
                  CuPtr{Cint}, CuPtr{Cint}, pruneInfo_t, CuPtr{Cvoid}),
                 handle, m, n, nnzA, descrA, csrSortedValA, csrSortedRowPtrA,
                 csrSortedColIndA, percentage, descrC, csrSortedValC, csrSortedRowPtrC,
                 csrSortedColIndC, info, pBuffer)
end

function cusparseCsr2cscEx2(handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, cscVal,
                            cscColPtr, cscRowInd, valType, copyValues, idxBase, alg, buffer)
    @check @runtime_ccall((:cusparseCsr2cscEx2, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, Cint, CuPtr{Cvoid}, CuPtr{Cint},
                  CuPtr{Cint}, CuPtr{Cvoid}, CuPtr{Cint}, CuPtr{Cint}, cudaDataType,
                  cusparseAction_t, cusparseIndexBase_t, cusparseCsr2CscAlg_t, CuPtr{Cvoid}),
                 handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, cscVal, cscColPtr,
                 cscRowInd, valType, copyValues, idxBase, alg, buffer)
end

function cusparseCsr2cscEx2_bufferSize(handle, m, n, nnz, csrVal, csrRowPtr, csrColInd,
                                       cscVal, cscColPtr, cscRowInd, valType, copyValues,
                                       idxBase, alg, bufferSize)
    @check @runtime_ccall((:cusparseCsr2cscEx2_bufferSize, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, Cint, Cint, Cint, CuPtr{Cvoid}, CuPtr{Cint},
                  CuPtr{Cint}, CuPtr{Cvoid}, CuPtr{Cint}, CuPtr{Cint}, cudaDataType,
                  cusparseAction_t, cusparseIndexBase_t, cusparseCsr2CscAlg_t, Ptr{Csize_t}),
                 handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, cscVal, cscColPtr,
                 cscRowInd, valType, copyValues, idxBase, alg, bufferSize)
end

function cusparseCreateSpVec(spVecDescr, size, nnz, indices, values, idxType, idxBase,
                             valueType)
    @check @runtime_ccall((:cusparseCreateSpVec, libcusparse()), cusparseStatus_t,
                 (Ptr{cusparseSpVecDescr_t}, Int64, Int64, CuPtr{Cvoid}, CuPtr{Cvoid},
                  cusparseIndexType_t, cusparseIndexBase_t, cudaDataType),
                 spVecDescr, size, nnz, indices, values, idxType, idxBase, valueType)
end

function cusparseDestroySpVec(spVecDescr)
    @check @runtime_ccall((:cusparseDestroySpVec, libcusparse()), cusparseStatus_t,
                 (cusparseSpVecDescr_t,),
                 spVecDescr)
end

function cusparseSpVecGet(spVecDescr, size, nnz, indices, values, idxType, idxBase,
                          valueType)
    @check @runtime_ccall((:cusparseSpVecGet, libcusparse()), cusparseStatus_t,
                 (cusparseSpVecDescr_t, Ptr{Int64}, Ptr{Int64}, CuPtr{Ptr{Cvoid}},
                  CuPtr{Ptr{Cvoid}}, Ptr{cusparseIndexType_t}, Ptr{cusparseIndexBase_t},
                  Ptr{cudaDataType}),
                 spVecDescr, size, nnz, indices, values, idxType, idxBase, valueType)
end

function cusparseSpVecGetIndexBase(spVecDescr, idxBase)
    @check @runtime_ccall((:cusparseSpVecGetIndexBase, libcusparse()), cusparseStatus_t,
                 (cusparseSpVecDescr_t, Ptr{cusparseIndexBase_t}),
                 spVecDescr, idxBase)
end

function cusparseSpVecGetValues(spVecDescr, values)
    @check @runtime_ccall((:cusparseSpVecGetValues, libcusparse()), cusparseStatus_t,
                 (cusparseSpVecDescr_t, CuPtr{Ptr{Cvoid}}),
                 spVecDescr, values)
end

function cusparseSpVecSetValues(spVecDescr, values)
    @check @runtime_ccall((:cusparseSpVecSetValues, libcusparse()), cusparseStatus_t,
                 (cusparseSpVecDescr_t, CuPtr{Cvoid}),
                 spVecDescr, values)
end

function cusparseCreateDnVec(dnVecDescr, size, values, valueType)
    @check @runtime_ccall((:cusparseCreateDnVec, libcusparse()), cusparseStatus_t,
                 (Ptr{cusparseDnVecDescr_t}, Int64, CuPtr{Cvoid}, cudaDataType),
                 dnVecDescr, size, values, valueType)
end

function cusparseDestroyDnVec(dnVecDescr)
    @check @runtime_ccall((:cusparseDestroyDnVec, libcusparse()), cusparseStatus_t,
                 (cusparseDnVecDescr_t,),
                 dnVecDescr)
end

function cusparseDnVecGet(dnVecDescr, size, values, valueType)
    @check @runtime_ccall((:cusparseDnVecGet, libcusparse()), cusparseStatus_t,
                 (cusparseDnVecDescr_t, Ptr{Int64}, CuPtr{Ptr{Cvoid}}, Ptr{cudaDataType}),
                 dnVecDescr, size, values, valueType)
end

function cusparseDnVecGetValues(dnVecDescr, values)
    @check @runtime_ccall((:cusparseDnVecGetValues, libcusparse()), cusparseStatus_t,
                 (cusparseDnVecDescr_t, CuPtr{Ptr{Cvoid}}),
                 dnVecDescr, values)
end

function cusparseDnVecSetValues(dnVecDescr, values)
    @check @runtime_ccall((:cusparseDnVecSetValues, libcusparse()), cusparseStatus_t,
                 (cusparseDnVecDescr_t, CuPtr{Cvoid}),
                 dnVecDescr, values)
end

function cusparseCreateCoo(spMatDescr, rows, cols, nnz, cooRowInd, cooColInd, cooValues,
                           cooIdxType, idxBase, valueType)
    @check @runtime_ccall((:cusparseCreateCoo, libcusparse()), cusparseStatus_t,
                 (Ptr{cusparseSpMatDescr_t}, Int64, Int64, Int64, CuPtr{Cvoid},
                  CuPtr{Cvoid}, CuPtr{Cvoid}, cusparseIndexType_t, cusparseIndexBase_t,
                  cudaDataType),
                 spMatDescr, rows, cols, nnz, cooRowInd, cooColInd, cooValues, cooIdxType,
                 idxBase, valueType)
end

function cusparseCreateCsr(spMatDescr, rows, cols, nnz, csrRowOffsets, csrColInd,
                           csrValues, csrRowOffsetsType, csrColIndType, idxBase, valueType)
    @check @runtime_ccall((:cusparseCreateCsr, libcusparse()), cusparseStatus_t,
                 (Ptr{cusparseSpMatDescr_t}, Int64, Int64, Int64, CuPtr{Cvoid},
                  CuPtr{Cvoid}, CuPtr{Cvoid}, cusparseIndexType_t, cusparseIndexType_t,
                  cusparseIndexBase_t, cudaDataType),
                 spMatDescr, rows, cols, nnz, csrRowOffsets, csrColInd, csrValues,
                 csrRowOffsetsType, csrColIndType, idxBase, valueType)
end

function cusparseCreateCooAoS(spMatDescr, rows, cols, nnz, cooInd, cooValues, cooIdxType,
                              idxBase, valueType)
    @check @runtime_ccall((:cusparseCreateCooAoS, libcusparse()), cusparseStatus_t,
                 (Ptr{cusparseSpMatDescr_t}, Int64, Int64, Int64, CuPtr{Cvoid},
                  CuPtr{Cvoid}, cusparseIndexType_t, cusparseIndexBase_t, cudaDataType),
                 spMatDescr, rows, cols, nnz, cooInd, cooValues, cooIdxType, idxBase,
                 valueType)
end

function cusparseDestroySpMat(spMatDescr)
    @check @runtime_ccall((:cusparseDestroySpMat, libcusparse()), cusparseStatus_t,
                 (cusparseSpMatDescr_t,),
                 spMatDescr)
end

function cusparseCooGet(spMatDescr, rows, cols, nnz, cooRowInd, cooColInd, cooValues,
                        idxType, idxBase, valueType)
    @check @runtime_ccall((:cusparseCooGet, libcusparse()), cusparseStatus_t,
                 (cusparseSpMatDescr_t, Ptr{Int64}, Ptr{Int64}, Ptr{Int64},
                  CuPtr{Ptr{Cvoid}}, CuPtr{Ptr{Cvoid}}, CuPtr{Ptr{Cvoid}},
                  Ptr{cusparseIndexType_t}, Ptr{cusparseIndexBase_t}, Ptr{cudaDataType}),
                 spMatDescr, rows, cols, nnz, cooRowInd, cooColInd, cooValues, idxType,
                 idxBase, valueType)
end

function cusparseCooAoSGet(spMatDescr, rows, cols, nnz, cooInd, cooValues, idxType,
                           idxBase, valueType)
    @check @runtime_ccall((:cusparseCooAoSGet, libcusparse()), cusparseStatus_t,
                 (cusparseSpMatDescr_t, Ptr{Int64}, Ptr{Int64}, Ptr{Int64},
                  CuPtr{Ptr{Cvoid}}, CuPtr{Ptr{Cvoid}}, Ptr{cusparseIndexType_t},
                  Ptr{cusparseIndexBase_t}, Ptr{cudaDataType}),
                 spMatDescr, rows, cols, nnz, cooInd, cooValues, idxType, idxBase,
                 valueType)
end

function cusparseCsrGet(spMatDescr, rows, cols, nnz, csrRowOffsets, csrColInd, csrValues,
                        csrRowOffsetsType, csrColIndType, idxBase, valueType)
    @check @runtime_ccall((:cusparseCsrGet, libcusparse()), cusparseStatus_t,
                 (cusparseSpMatDescr_t, Ptr{Int64}, Ptr{Int64}, Ptr{Int64},
                  CuPtr{Ptr{Cvoid}}, CuPtr{Ptr{Cvoid}}, CuPtr{Ptr{Cvoid}},
                  Ptr{cusparseIndexType_t}, Ptr{cusparseIndexType_t},
                  Ptr{cusparseIndexBase_t}, Ptr{cudaDataType}),
                 spMatDescr, rows, cols, nnz, csrRowOffsets, csrColInd, csrValues,
                 csrRowOffsetsType, csrColIndType, idxBase, valueType)
end

function cusparseSpMatGetFormat(spMatDescr, format)
    @check @runtime_ccall((:cusparseSpMatGetFormat, libcusparse()), cusparseStatus_t,
                 (cusparseSpMatDescr_t, Ptr{cusparseFormat_t}),
                 spMatDescr, format)
end

function cusparseSpMatGetIndexBase(spMatDescr, idxBase)
    @check @runtime_ccall((:cusparseSpMatGetIndexBase, libcusparse()), cusparseStatus_t,
                 (cusparseSpMatDescr_t, Ptr{cusparseIndexBase_t}),
                 spMatDescr, idxBase)
end

function cusparseSpMatGetValues(spMatDescr, values)
    @check @runtime_ccall((:cusparseSpMatGetValues, libcusparse()), cusparseStatus_t,
                 (cusparseSpMatDescr_t, CuPtr{Ptr{Cvoid}}),
                 spMatDescr, values)
end

function cusparseSpMatSetValues(spMatDescr, values)
    @check @runtime_ccall((:cusparseSpMatSetValues, libcusparse()), cusparseStatus_t,
                 (cusparseSpMatDescr_t, CuPtr{Cvoid}),
                 spMatDescr, values)
end

function cusparseSpMatSetStridedBatch(spMatDescr, batchCount)
    @check @runtime_ccall((:cusparseSpMatSetStridedBatch, libcusparse()), cusparseStatus_t,
                 (cusparseSpMatDescr_t, Cint),
                 spMatDescr, batchCount)
end

function cusparseSpMatGetStridedBatch(spMatDescr, batchCount)
    @check @runtime_ccall((:cusparseSpMatGetStridedBatch, libcusparse()), cusparseStatus_t,
                 (cusparseSpMatDescr_t, Ptr{Cint}),
                 spMatDescr, batchCount)
end

function cusparseCreateDnMat(dnMatDescr, rows, cols, ld, values, valueType, order)
    @check @runtime_ccall((:cusparseCreateDnMat, libcusparse()), cusparseStatus_t,
                 (Ptr{cusparseDnMatDescr_t}, Int64, Int64, Int64, CuPtr{Cvoid},
                  cudaDataType, cusparseOrder_t),
                 dnMatDescr, rows, cols, ld, values, valueType, order)
end

function cusparseDestroyDnMat(dnMatDescr)
    @check @runtime_ccall((:cusparseDestroyDnMat, libcusparse()), cusparseStatus_t,
                 (cusparseDnMatDescr_t,),
                 dnMatDescr)
end

function cusparseDnMatGet(dnMatDescr, rows, cols, ld, values, type, order)
    @check @runtime_ccall((:cusparseDnMatGet, libcusparse()), cusparseStatus_t,
                 (cusparseDnMatDescr_t, Ptr{Int64}, Ptr{Int64}, Ptr{Int64},
                  CuPtr{Ptr{Cvoid}}, Ptr{cudaDataType}, Ptr{cusparseOrder_t}),
                 dnMatDescr, rows, cols, ld, values, type, order)
end

function cusparseDnMatGetValues(dnMatDescr, values)
    @check @runtime_ccall((:cusparseDnMatGetValues, libcusparse()), cusparseStatus_t,
                 (cusparseDnMatDescr_t, CuPtr{Ptr{Cvoid}}),
                 dnMatDescr, values)
end

function cusparseDnMatSetValues(dnMatDescr, values)
    @check @runtime_ccall((:cusparseDnMatSetValues, libcusparse()), cusparseStatus_t,
                 (cusparseDnMatDescr_t, CuPtr{Cvoid}),
                 dnMatDescr, values)
end

function cusparseDnMatSetStridedBatch(dnMatDescr, batchCount, batchStride)
    @check @runtime_ccall((:cusparseDnMatSetStridedBatch, libcusparse()), cusparseStatus_t,
                 (cusparseDnMatDescr_t, Cint, Int64),
                 dnMatDescr, batchCount, batchStride)
end

function cusparseDnMatGetStridedBatch(dnMatDescr, batchCount, batchStride)
    @check @runtime_ccall((:cusparseDnMatGetStridedBatch, libcusparse()), cusparseStatus_t,
                 (cusparseDnMatDescr_t, Ptr{Cint}, Ptr{Int64}),
                 dnMatDescr, batchCount, batchStride)
end

function cusparseSpVV(handle, opX, vecX, vecY, result, computeType, externalBuffer)
    @check @runtime_ccall((:cusparseSpVV, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseOperation_t, cusparseSpVecDescr_t,
                  cusparseDnVecDescr_t, PtrOrCuPtr{Cvoid}, cudaDataType, CuPtr{Cvoid}),
                 handle, opX, vecX, vecY, result, computeType, externalBuffer)
end

function cusparseSpVV_bufferSize(handle, opX, vecX, vecY, result, computeType, bufferSize)
    @check @runtime_ccall((:cusparseSpVV_bufferSize, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseOperation_t, cusparseSpVecDescr_t,
                  cusparseDnVecDescr_t, PtrOrCuPtr{Cvoid}, cudaDataType, Ptr{Csize_t}),
                 handle, opX, vecX, vecY, result, computeType, bufferSize)
end

function cusparseSpMV(handle, opA, alpha, matA, vecX, beta, vecY, computeType, alg,
                      externalBuffer)
    @check @runtime_ccall((:cusparseSpMV, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseOperation_t, PtrOrCuPtr{Cvoid},
                  cusparseSpMatDescr_t, cusparseDnVecDescr_t, PtrOrCuPtr{Cvoid},
                  cusparseDnVecDescr_t, cudaDataType, cusparseSpMVAlg_t, CuPtr{Cvoid}),
                 handle, opA, alpha, matA, vecX, beta, vecY, computeType, alg,
                 externalBuffer)
end

function cusparseSpMV_bufferSize(handle, opA, alpha, matA, vecX, beta, vecY, computeType,
                                 alg, bufferSize)
    @check @runtime_ccall((:cusparseSpMV_bufferSize, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseOperation_t, Ptr{Cvoid}, cusparseSpMatDescr_t,
                  cusparseDnVecDescr_t, Ptr{Cvoid}, cusparseDnVecDescr_t, cudaDataType,
                  cusparseSpMVAlg_t, Ptr{Csize_t}),
                 handle, opA, alpha, matA, vecX, beta, vecY, computeType, alg, bufferSize)
end

function cusparseSpMM(handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg,
                      externalBuffer)
    @check @runtime_ccall((:cusparseSpMM, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseOperation_t, cusparseOperation_t,
                  PtrOrCuPtr{Cvoid}, cusparseSpMatDescr_t, cusparseDnMatDescr_t,
                  PtrOrCuPtr{Cvoid}, cusparseDnMatDescr_t, cudaDataType, cusparseSpMMAlg_t,
                  CuPtr{Cvoid}),
                 handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg,
                 externalBuffer)
end

function cusparseSpMM_bufferSize(handle, opA, opB, alpha, matA, matB, beta, matC,
                                 computeType, alg, bufferSize)
    @check @runtime_ccall((:cusparseSpMM_bufferSize, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, Ptr{Cvoid},
                  cusparseSpMatDescr_t, cusparseDnMatDescr_t, Ptr{Cvoid},
                  cusparseDnMatDescr_t, cudaDataType, cusparseSpMMAlg_t, Ptr{Csize_t}),
                 handle, opA, opB, alpha, matA, matB, beta, matC, computeType, alg,
                 bufferSize)
end

function cusparseConstrainedGeMM(handle, opA, opB, alpha, matA, matB, beta, matC,
                                 computeType, externalBuffer)
    @check @runtime_ccall((:cusparseConstrainedGeMM, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseOperation_t, cusparseOperation_t,
                  PtrOrCuPtr{Cvoid}, cusparseDnMatDescr_t, cusparseDnMatDescr_t,
                  PtrOrCuPtr{Cvoid}, cusparseSpMatDescr_t, cudaDataType, CuPtr{Cvoid}),
                 handle, opA, opB, alpha, matA, matB, beta, matC, computeType,
                 externalBuffer)
end

function cusparseConstrainedGeMM_bufferSize(handle, opA, opB, alpha, matA, matB, beta,
                                            matC, computeType, bufferSize)
    @check @runtime_ccall((:cusparseConstrainedGeMM_bufferSize, libcusparse()), cusparseStatus_t,
                 (cusparseHandle_t, cusparseOperation_t, cusparseOperation_t, Ptr{Cvoid},
                  cusparseDnMatDescr_t, cusparseDnMatDescr_t, Ptr{Cvoid},
                  cusparseSpMatDescr_t, cudaDataType, Ptr{Csize_t}),
                 handle, opA, opB, alpha, matA, matB, beta, matC, computeType, bufferSize)
end
