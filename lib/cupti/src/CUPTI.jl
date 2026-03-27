module CUPTI

using GPUToolbox

using CUDACore
using CUDACore: retry_reclaim, initialize_context
using CUDACore: CUuuid, CUcontext, CUstream, CUdevice, CUdevice_attribute, CUstreamAttrID,
              CUgraph, CUgraphNode, CUgraphNodeType, CUgraphExec, CUaccessPolicyWindow,
              CUstreamAttrValue

if CUDACore.local_toolkit
    using CUDA_Runtime_Discovery
else
    import CUDA_Runtime_jll
end

using CEnum: @cenum


# core library
include("libcupti.jl")

include("error.jl")
include("wrappers.jl")


function __init__()
    CUDACore.functional() || return

    # find the library
    global libcupti
    if CUDACore.local_toolkit
        dirs = CUDA_Runtime_Discovery.find_toolkit()
        path = CUDA_Runtime_Discovery.get_library(dirs, "cupti"; optional=true)
        if path === nothing
            return
        end
        libcupti = path
    else
        libcupti = CUDA_Runtime_jll.libcupti
    end
end

end
