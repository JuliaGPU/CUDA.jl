# cuSPARSE functions for managing the library

function cusparseCreate()
    handle = Ref{cusparseHandle_t}()
    @check unsafe_cusparseCreate(handle) CUSPARSE_STATUS_NOT_INITIALIZED
    handle[]
end

@memoize function cusparseGetProperty(property::libraryPropertyType)
    value_ref = Ref{Cint}()
    cusparseGetProperty(property, value_ref)
    value_ref[]
end

@memoize version() = VersionNumber(cusparseGetProperty(CUDA.MAJOR_VERSION),
                                   cusparseGetProperty(CUDA.MINOR_VERSION),
                                   cusparseGetProperty(CUDA.PATCH_LEVEL))
