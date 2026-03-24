import KernelAbstractions
import KernelAbstractions as KA

include(joinpath(dirname(pathof(KernelAbstractions)), "..", "test", "testsuite.jl"))

ka_skip_tests = Set{String}(["sparse"])
Testsuite.testsuite(()->CUDABackend(false, false), "CUDA", CUDACore, CuArray, CuDeviceArray;
                    skip_tests=ka_skip_tests)
for (PreferBlocks, AlwaysInline) in Iterators.product((true, false), (true, false))
    Testsuite.unittest_testsuite(()->CUDABackend(PreferBlocks, AlwaysInline), "CUDA", CUDACore, CuDeviceArray;
                                 skip_tests=ka_skip_tests)
end

@testset "KA.functional" begin
    @test KA.functional(CUDABackend()) == CUDACore.functional()
end
