module cuFFT

using CUDACore
using GPUToolbox
using CUDACore: CUstream, cuComplex, cuDoubleComplex, cudaDataType, libraryPropertyType
using CUDACore: unsafe_free!, retry_reclaim, initialize_context

using CEnum: @cenum

using Reexport: @reexport

if CUDACore.local_toolkit
    using CUDA_Runtime_Discovery
else
    import CUDA_Runtime_jll
end


@public functional

const _initialized = Ref{Bool}(false)
functional() = _initialized[]

# core library
include("libcufft.jl")

# low-level wrappers
include("error.jl")
include("util.jl")
include("wrappers.jl")

# high-level integrations
include("fft.jl")


function __init__()
    precompiling = ccall(:jl_generating_output, Cint, ()) != 0

    CUDACore.functional() || return

    # find the library
    global libcufft
    if CUDACore.local_toolkit
        dirs = CUDA_Runtime_Discovery.find_toolkit()
        path = CUDA_Runtime_Discovery.get_library(dirs, "cufft"; optional=true)
        if path === nothing
            precompiling || @error "cuFFT is not available on your system (looked in $(join(dirs, ", ")))"
            return
        end
        libcufft = path
    else
        libcufft = CUDA_Runtime_jll.libcufft
    end

    _initialized[] = true
end

# deprecated binding for backwards compatibility
Base.@deprecate_binding CUFFT cuFFT false

end
