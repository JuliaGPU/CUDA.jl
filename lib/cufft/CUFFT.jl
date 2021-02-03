module CUFFT

using ..APIUtils

using ..CUDA
using ..CUDA: CUstream, cuComplex, cuDoubleComplex, libraryPropertyType
import ..CUDA: libcufft, unsafe_free!, @retry_reclaim

using CEnum

using Reexport

using Memoize


# core library
include("libcufft_common.jl")
include("error.jl")
include("libcufft.jl")

# low-level wrappers
include("util.jl")
include("wrappers.jl")

# high-level integrations
include("fft.jl")

function set_stream(s)
    # CUFFT associates streams to plan objects
end

end
