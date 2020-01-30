module CUFFT

using CUDAapi

using ..CuArrays
import ..CuArrays: unsafe_free!

using CUDAdrv
using CUDAdrv: CUstream

using CUDAnative

using CEnum

using Reexport

const libcufft = Ref("libcufft")

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
