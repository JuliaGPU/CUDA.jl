using CUDAapi

config_path = joinpath(@__DIR__, "ext.jl")
config = Dict()

config[:libcublas] = CUDAapi.find_library("cublas", CUDAapi.find_toolkit())

open(config_path, "w") do io
  for (key,val) in config
    println(io, "const $key = $(repr(val))")
  end
end
