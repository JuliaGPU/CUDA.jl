# cuSPARSE functions for managing the library

function cusparseCreate()
    handle = Ref{cusparseHandle_t}()
    cusparseCreate(handle)
    handle[]
end

function cusparseGetProperty(property::libraryPropertyType)
    value_ref = Ref{Cint}()
    cusparseGetProperty(property, value_ref)
    value_ref[]
end

version() = VersionNumber(cusparseGetProperty(CUDA.MAJOR_VERSION),
                          cusparseGetProperty(CUDA.MINOR_VERSION),
                          cusparseGetProperty(CUDA.PATCH_LEVEL))
