import KernelInterface
using CUDACore

include(joinpath(dirname(pathof(KernelInterface)), "..", "test", "testsuite.jl"))

Testsuite.testsuite(CUDABackend, "CUDACore", CUDACore, CuArray, CUDACore.CuDeviceArray)
