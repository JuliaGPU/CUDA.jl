module CUSPARSE

using ..CuArrays
using ..CuArrays: libcusparse, active_context, unsafe_free!

using CUDAapi

using CUDAdrv
import CUDAdrv: CuStream_t

import CUDAnative

using CEnum

const SparseChar = Char

include("libcusparse_common.jl")
include("error.jl")

version() = VersionNumber(cusparseGetProperty(CUDAapi.MAJOR_VERSION),
                          cusparseGetProperty(CUDAapi.MINOR_VERSION),
                          cusparseGetProperty(CUDAapi.PATCH_LEVEL))

include("libcusparse.jl")
include("array.jl")
include("util.jl")
include("wrappers.jl")
include("linalg.jl")

const _handles = Dict{CuContext,cusparseHandle_t}()
const _handle = Ref{cusparseHandle_t}()

function handle()
    if _handle[] == C_NULL
        CUDAnative.maybe_initialize("CUSPARSE")
        _handle[] = get!(_handles, active_context[]) do
            context = active_context[]
            handle = cusparseCreate()
            atexit(()->CUDAdrv.isvalid(context) && cusparseDestroy(handle))
            handle
        end
    end

    return _handle[]
end

end
