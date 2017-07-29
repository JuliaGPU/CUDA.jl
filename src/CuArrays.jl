module CuArrays

using CUDAdrv, CUDAnative

export CuArray

include("array.jl")
include("slow.jl")

end # module
