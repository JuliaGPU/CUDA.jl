module CUFILE

using ..APIUtils

using ..CUDA
using ..CUDA: CUresult
using ..CUDA: initialize_context

using CEnum: @cenum

const libcufile = "/home/tim/Julia/pkg/CUDA/res/gds/usr/lib/libcufile.so"

# core library
include("libcufile_common.jl")
include("error.jl")
include("libcufile.jl")

end
