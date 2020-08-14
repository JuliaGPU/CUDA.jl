# cuSPARSE functions for managing the library

function cusparseCreate()
    handle = Ref{cusparseHandle_t}()
    res = @retry_reclaim err->isequal(err, CUSPARSE_STATUS_ALLOC_FAILED) ||
                              isequal(err, CUSPARSE_STATUS_NOT_INITIALIZED) begin
        unsafe_cusparseCreate(handle)
    end
    if res != CUSPARSE_STATUS_SUCCESS
         throw_api_error(res)
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
