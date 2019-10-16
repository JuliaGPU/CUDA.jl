module NVPerf

using CUDAapi

using CEnum

const libnvperf = Ref("libnvperf")

# core library
include("libnvperf_common.jl")
include("error.jl")
include("libnvperf.jl")

end
