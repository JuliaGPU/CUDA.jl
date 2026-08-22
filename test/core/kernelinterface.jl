import KernelInterface
using CUDACore.CUDAInterface

include(joinpath(dirname(pathof(KernelInterface)), "..", "test", "testsuite.jl"))

Testsuite.testsuite(CUDAInterface.CUDABackend, "CUDACore", CUDACore, CuArray, CUDACore.CuDeviceArray)
