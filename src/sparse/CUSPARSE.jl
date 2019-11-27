module CUSPARSE

using ..CuArrays
using ..CuArrays: active_context, unsafe_free!, @argout, @workspace

using CUDAapi

using CUDAdrv
using CUDAdrv: CUstream

import CUDAnative

using CEnum

const SparseChar = Char

const libcusparse = if Sys.iswindows()
    # no ccall by soname, we need the filename
    # NOTE: we discover the full path here, while only the wordsize and toolkit versions
    #       would have been enough to construct "cusparse64_10.dll"
    toolkit = find_toolkit()
    path = find_cuda_library("cusparse", toolkit)
    if path === nothing
        error("Could not find libcusparse")
    end
    basename(path)
else
    # ccall by soname; CuArrays.__init__ will have populated Libdl.DL_LOAD_PATH
    "libcusparse"
end

# core library
include("libcusparse_common.jl")
include("error.jl")
include("libcusparse.jl")

# low-level wrappers
include("array.jl")
include("util.jl")
include("wrappers.jl")

# high-level integrations
include("interfaces.jl")

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
