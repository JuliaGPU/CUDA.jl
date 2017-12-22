using CUDAapi

using Compat
using Compat.Test


## logging

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

macro test_something(ex...)
    quote
        rv = $(ex...)
        @test rv != nothing
        rv
    end
end

toolkit = @test_something find_toolkit()
toolkit_version = find_toolkit_version(toolkit)

if haskey(ENV, "CI")
    find_driver()
else
    @test_something find_driver()
end

@test_something find_cuda_binary("nvcc", toolkit)
@test_something find_cuda_library("cudart", toolkit)
@test_something find_host_compiler()
@test_something find_host_compiler(toolkit_version)
@test_something find_toolchain(toolkit)
@test_something find_toolchain(toolkit, toolkit_version)
