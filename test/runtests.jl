using CUDAdrv
using Base.Test

@test devcount() > 0

include("core.jl")
