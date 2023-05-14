# cuSPARSE functions for managing the library

function cusparseCreate()
    handle = Ref{cusparseHandle_t}()
    check(CUSPARSE_STATUS_NOT_INITIALIZED) do
        unsafe_cusparseCreate(handle)
    end
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
