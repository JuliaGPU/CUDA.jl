module CUFFT

using ..APIUtils

using ..CUDA_Runtime

using ..CUDA
using ..CUDA: CUstream, cuComplex, cuDoubleComplex, libraryPropertyType
using ..CUDA: unsafe_free!, retry_reclaim, initialize_context

using CEnum: @cenum

using Reexport: @reexport


# core library
include("libcufft.jl")

# low-level wrappers
include("error.jl")
include("util.jl")
include("wrappers.jl")

# high-level integrations
include("fft.jl")

end
