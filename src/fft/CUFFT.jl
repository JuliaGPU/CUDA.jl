module CUFFT

using CUDAapi

using ..CuArrays
using ..CuArrays: libcufft
import ..CuArrays: unsafe_free!

using CUDAdrv
using CUDAdrv: CuStream_t

using CEnum

include("libcufft_common.jl")
include("error.jl")

version() = VersionNumber(cufftGetProperty(CUDAapi.MAJOR_VERSION),
                          cufftGetProperty(CUDAapi.MINOR_VERSION),
                          cufftGetProperty(CUDAapi.PATCH_LEVEL))

include("libcufft.jl")

include("util.jl")
include("wrappers.jl")
include("fft.jl")

end
