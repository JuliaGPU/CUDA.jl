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

"""
    cuSPARSE.version() -> VersionNumber

The version of the cuSPARSE library in use, which differs from the version of the CUDA
toolkit.
"""
version() = VersionNumber(cusparseGetProperty(CUDACore.MAJOR_VERSION),
                          cusparseGetProperty(CUDACore.MINOR_VERSION),
                          cusparseGetProperty(CUDACore.PATCH_LEVEL))
