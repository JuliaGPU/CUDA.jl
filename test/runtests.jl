using CUDAapi
using Base.Test


## logging

@trace("")

@debug("")

@logging_ccall(:time, :time, Cint, ())

logging_run(`true`)


## properties

CUDAapi.gcc_for_cuda(v"8.0")
CUDAapi.devices_for_cuda(v"8.0")
CUDAapi.devices_for_llvm(v"5.0")


## discovery

toolkit = find_toolkit()
find_driver()
find_binary("true")
find_binary("nvcc", toolkit)
find_binary("nvcc", [toolkit])
find_library("c")
find_library("cudart", toolkit)
find_library("cudart", [toolkit])
find_toolchain(toolkit)
