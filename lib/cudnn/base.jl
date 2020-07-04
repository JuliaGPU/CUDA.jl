function cudnnCreate()
    handle_ref = Ref{cudnnHandle_t}()
    res = @retry_reclaim err->isequal(err, CUDNN_STATUS_ALLOC_FAILED) ||
                              isequal(err, CUDNN_STATUS_NOT_INITIALIZED) ||
                              isequal(err, CUDNN_STATUS_INTERNAL_ERROR) begin
        unsafe_cudnnCreate(handle_ref)
    end
    if res != CUDNN_STATUS_SUCCESS
         throw_api_error(res)
    end
    return handle_ref[]
end

function cudnnGetProperty(property::CUDA.libraryPropertyType)
  value_ref = Ref{Cint}()
  cudnnGetProperty(property, value_ref)
  value_ref[]
end

version() = VersionNumber(cudnnGetProperty(CUDA.MAJOR_VERSION),
                          cudnnGetProperty(CUDA.MINOR_VERSION),
                          cudnnGetProperty(CUDA.PATCH_LEVEL))

function cuda_version()
  ver = cudnnGetCudartVersion()
  major, ver = divrem(ver, 1000)
  minor, patch = divrem(ver, 10)

  VersionNumber(major, minor, patch)
end
