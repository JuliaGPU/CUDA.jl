module CUSPARSE

using ..CuArrays
using ..CuArrays: active_context, unsafe_free!, @argout, @workspace

using CUDAapi

using CUDAdrv
using CUDAdrv: CUstream

import CUDAnative

using CEnum

const SparseChar = Char

const libcusparse = Ref("libcusparse")

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
