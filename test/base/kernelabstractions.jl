import KernelAbstractions
include(joinpath(dirname(pathof(KernelAbstractions)), "..", "test", "testsuite.jl"))

Testsuite.testsuite(()->CUDABackend(false, false), "CUDA", CUDA, CuArray, CuDeviceArray)
for (PreferBlocks, AlwaysInline) in Iterators.product((true, false), (true, false))
    Testsuite.unittest_testsuite(()->CUDABackend(PreferBlocks, AlwaysInline), "CUDA", CUDA, CuDeviceArray)
end
