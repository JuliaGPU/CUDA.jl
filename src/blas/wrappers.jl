# low-level wrappers

function cublasCreate_v2()
  handle = Ref{cublasHandle_t}()
  cublasCreate_v2(handle)
  handle[]
end

function cublasXtCreate()
  handle = Ref{cublasXtHandle_t}()
  cublasXtCreate(handle)
  handle[]
end

function cublasXtGetBlockDim(handle)
  bd = Ref{Int}()
  cublasXtGetBlockDim(handle, bd)
  bd[]
end

function cublasXtGetPinningMemMode(handle)
  mm = Ref{cublasXtPinningMemMode_t}()
  cublasXtGetPinningMemMode(handle, mm)
  mm[]
end

function cublasGetVersion_v2(handle)
  version = Ref{Cint}()
  cublasGetVersion_v2(handle, version)
  version[]
end

function cublasGetProperty(property::CUDAapi.libraryPropertyType)
  value_ref = Ref{Cint}()
  cublasGetProperty(property, value_ref)
  value_ref[]
end
