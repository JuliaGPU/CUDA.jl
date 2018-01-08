using CUDAapi

config_path = joinpath(@__DIR__, "ext.jl")
config = Dict()

toolkit = CUDAapi.find_toolkit()

config[:libcublas] = CUDAapi.find_cuda_library("cublas", toolkit)
config[:libcusolver] = CUDAapi.find_cuda_library("cusolver", toolkit)

config[:libcudnn] = CUDAapi.find_cuda_library("cudnn", toolkit)
if config[:libcudnn] == nothing
  warn("could not find CUDNN, its functionality will be unavailable")
end

open(config_path, "w") do io
  for (key,val) in config
    println(io, "const $key = $(repr(val))")
  end
end
