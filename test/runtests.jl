using CUDAapi

using Compat
using Compat.Test


## logging

@trace("test")

@debug("test")


## properties

CUDAapi.gcc_for_cuda(v"8.0")
CUDAapi.devices_for_cuda(v"8.0")
CUDAapi.devices_for_llvm(v"5.0")
CUDAapi.isas_for_cuda(v"8.0")
CUDAapi.isas_for_llvm(v"5.0")


## discovery

toolkit = find_toolkit()
find_toolkit_version(toolkit)
find_driver()
find_binary(Compat.Sys.iswindows() ? "CHKDSK" : "true")
find_binary(CUDAapi.nvcc, toolkit)
find_binary(CUDAapi.nvcc, [toolkit])
find_library(Compat.Sys.iswindows() ? "NTDLL" : "c")
find_library([Compat.Sys.iswindows() ? "NTDLL" : "c"])
find_library(CUDAapi.libcudart, toolkit)
find_library(CUDAapi.libcudart, [toolkit])
find_toolchain(toolkit)
