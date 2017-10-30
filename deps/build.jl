using CUDAapi

config_path = joinpath(@__DIR__, "ext.jl")
config = Dict()

toolkit = CUDAapi.find_toolkit()

config[:libcublas] = CUDAapi.find_library("cublas", toolkit)
config[:libcusolver] = CUDAapi.find_library("cusolver", toolkit)

config[:libcudnn] = try
  CUDAapi.find_library("cudnn", toolkit)
catch e
  warn("No CUDNN available: $(sprint(showerror, e))")
end

open(config_path, "w") do io
  for (key,val) in config
    println(io, "const $key = $(repr(val))")
  end
end
