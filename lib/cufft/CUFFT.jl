module CUFFT

using ..APIUtils

using ..CUDA
using ..CUDA: CUstream, cuComplex, cuDoubleComplex, libraryPropertyType
using ..CUDA: libcufft, unsafe_free!, @retry_reclaim, @context!

using CEnum: @cenum

using Reexport: @reexport


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
