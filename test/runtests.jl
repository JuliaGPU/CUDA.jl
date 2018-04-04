using CUDAapi

using Compat
using Compat.Test


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

macro test_something(ex...)
    quote
        rv = $(ex...)
        @test rv != nothing
        rv
    end
end

dirs = find_toolkit()
@test !isempty(dirs)
ver = find_toolkit_version(dirs)

@test_something find_cuda_binary("nvcc", dirs)
@test_something find_cuda_library("cudart", dirs)
@test_something find_libdevice([v"3.0"], dirs)
@test_something find_host_compiler()
@test_something find_host_compiler(ver)
@test_something find_toolchain(dirs)
@test_something find_toolchain(dirs, ver)
