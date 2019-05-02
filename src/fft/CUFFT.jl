module CUFFT

import CUDAapi

using ..CuArrays
using ..CuArrays: libcufft, configured

import AbstractFFTs: plan_fft, plan_fft!, plan_bfft, plan_bfft!,
    plan_rfft, plan_brfft, plan_inv, normalization, fft, bfft, ifft, rfft,
    Plan, ScaledPlan
import Base: show, *, convert, unsafe_convert, size, strides, ndims
import Base.Sys: WORD_SIZE

using LinearAlgebra
import LinearAlgebra: mul!

include("libcufft_types.jl")
include("error.jl")

include("libcufft.jl")
include("genericfft.jl")
include("fft.jl")
include("wrappers.jl")
include("highlevel.jl")

version() = VersionNumber(cufftGetProperty(CUDAapi.MAJOR_VERSION),
                          cufftGetProperty(CUDAapi.MINOR_VERSION),
                          cufftGetProperty(CUDAapi.PATCH_LEVEL))

end
