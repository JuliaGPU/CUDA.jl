using CUDAapi

using Compat
using Compat.Test


## logging

@trace("test")

@debug("test")


## properties

@test !CUDAapi.gcc_supported(v"5.0", v"5.5")
@test CUDAapi.gcc_supported(v"5.0", v"8.0")
CUDAapi.devices_for_cuda(v"8.0")
CUDAapi.devices_for_llvm(v"5.0")
CUDAapi.isas_for_cuda(v"8.0")
CUDAapi.isas_for_llvm(v"5.0")


## discovery

# generic
find_binary([Compat.Sys.iswindows() ? "CHKDSK" : "true"])
find_library([Compat.Sys.iswindows() ? "NTDLL" : "c"])

# CUDA

toolkit = find_cuda_toolkit()
toolkit_version = find_cuda_toolkit_version(toolkit)
# find_cuda_driver()
find_cuda_binary("nvcc", toolkit)
find_cuda_library("cudart", toolkit)
find_host_compiler()
find_host_compiler(toolkit_version)
find_toolchain(toolkit)
find_toolchain(toolkit, toolkit_version)
