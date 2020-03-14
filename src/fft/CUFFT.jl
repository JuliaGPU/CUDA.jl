module CUFFT

using CUDAapi

using ..CuArrays
import ..CuArrays: libcufft, unsafe_free!, @retry_reclaim

using CUDAdrv
using CUDAdrv: CUstream

using CUDAnative

using CEnum

using Reexport

# core library
include("libcufft_common.jl")
include("error.jl")
include("libcufft.jl")

# low-level wrappers
include("util.jl")
include("wrappers.jl")

# high-level integrations
include("fft.jl")

end
